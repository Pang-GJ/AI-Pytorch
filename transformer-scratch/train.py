import warnings
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from dataset import BilingualDataset, causal_mask
from model import build_transformer, Transformer
from config import get_config, get_weights_file_path


def greedy_decode(
    model: Transformer,
    encoder_input,
    encoder_mask,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len,
    device,
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(encoder_input, encoder_mask)
    # initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the decoder input
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        )
        # forward pass through the decoder
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        # get the next token
        prob = model.project(out[:, -1])
        # select the token with the highest probability (greedy decoding)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(encoder_input)
                .fill_(next_word.item())
                .to(device),
            ],
            dim=1,
        )
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def beam_search_decode(
    model: Transformer,
    beam_size,
    encoder_input,
    encoder_mask,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len,
    device,
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(encoder_input, encoder_mask)
    # initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

    # create a candidate list
    candidates = [(decoder_input, 1)]

    while True:
        # if a candidate has reached the maximum length,
        # it means we have run the decoding for a least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand in candidates]):
            break

        new_candidates = []
        for candidate, score in candidates:
            # not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue
            decoder_mask = (
                causal_mask(candidate.size(1)).type_as(encoder_mask).to(device)
            )
            # forward pass through the decoder
            out = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            # get the next token
            prob = model.project(out[:, -1])
            # beam search: select the top k tokens with the highest probability
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k tokens, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)
                new_candidates.append((new_candidate, score + token_prob))
        # sort the candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # select the top k candidates
        candidates = candidates[:beam_size]

        # if all candidates have reached the eos token, stop the search
        if all([cand[0][-1].item() == eos_idx for cand in candidates]):
            break
    # return the best candidate
    return candidates[0][0].squeeze()


def run_validation(
    model: Transformer,
    val_dataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0
    # size of the control window
    console_width = 80
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in val_dataLoader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"

            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print the output to the console
            print_msg("-" * console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, dataset, lang):
    # config["tokenizer_file"] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        # Build the tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    # keep 90% of the data for training, 10% for validation
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length of source sequence: {max_len_src}")
    print(f"Max length of target sequence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_size, vocab_tgt_size):
    model = build_transformer(
        vocab_src_size,
        vocab_tgt_size,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    # tensorboard
    writer = SummaryWriter(log_dir=config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    )

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch, 1, seq_len, seq_len)

            # run the tensor through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch, seq_len, d_model)
            proj_output = model.project(
                decoder_output
            )  # (batch, seq_len, vocab_tgt_size)

            label = batch["label"].to(device)  # (batch, seq_len)
            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            # show the loss in the progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # log the loss in tensorboard
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer=writer,
        )
        # save the model every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
