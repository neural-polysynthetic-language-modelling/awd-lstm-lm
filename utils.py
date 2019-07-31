import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(1) // bsz
    embed_size = data.size(0)
    print(embed_size)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(1, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches
    data = data.view(embed_size, bsz, -1).transpose(1,2).contiguous()
    print("Shape of data after view: " + str(data.shape))
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    embed_size = source.size(0)
    print(i)
    print(seq_len)
    data = source[:, i:i+seq_len]
    target = source[:, i+1:i+1+seq_len].view(embed_size, -1)
    return data, target
