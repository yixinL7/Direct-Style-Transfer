import torch

def generate(model, 
        input_ids,
        cur_len,
        max_length,
        pad_token_id,
        eos_token_ids,
        batch_size,
        approximate=False,
        style_token=None
    ):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    # current position / max lengths / length of generated sentences / unfinished sentences
    if approximate:
        unfinished_sents = style_token.new(batch_size).fill_(1)
        sent_lengths = style_token.new(batch_size).fill_(max_length)
    else:
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
    past = None
    start = True
    generated_logits = []
    if approximate:
        embed = torch.matmul(input_ids, model.transformer.wte.weight)
        embed = torch.cat((embed, model.transformer.wte(style_token).unsqueeze(1)), dim=1)
    while cur_len < max_length:
        if approximate and start:
            # model_inputs = model.prepare_inputs_for_generation(input_embeds=embed, past=past)
            outputs = model(inputs_embeds=embed, past=past)
        else:
            model_inputs = model.prepare_inputs_for_generation(input_ids, past=past)
            outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]
        generated_logits.append(next_token_logits)
        if model._do_output_past(outputs):
            past = outputs[1]
        next_token = torch.argmax(next_token_logits, dim=-1)
        # update generations and finished sentences
        if eos_token_ids is not None:
            # pad finished sentences if eos_token_ids exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token
        if approximate and start:
            input_ids = tokens_to_add.unsqueeze(-1)
        else:
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        if eos_token_ids is not None:
            for eos_token_id in eos_token_ids:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())
        cur_len = cur_len + 1
        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            # break
            pass
        start = False
    if approximate:
        decoded = input_ids
    else:
        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids
        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]
    return decoded, torch.stack(generated_logits, dim=1)

import torch

def approximate(model, input_ids):
    """ Approximate """
    embed = torch.matmul(input_ids, model.transformer.wte.weight)
    outputs = model(inputs_embeds=embed)
    return outputs