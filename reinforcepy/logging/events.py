def minibatch_end(step, values):
    return {"type": "Minibatch", "step": step, "values": values}


def epoch_end(step, values):
    return {"type": "Epoch", "step": step, "values": values}

