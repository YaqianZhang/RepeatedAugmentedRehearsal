import torch
import numpy as np


class Replay_times_update(object):
    def __init__(self, params):
        super().__init__()

    def update(self, buffer, x, y, **kwargs):
        batch_size = x.size(0)

        # add whatever still fits in the buffer
        place_left = max(0, buffer.buffer_img.size(0) - buffer.current_index)
        if place_left:
            offset = min(place_left, batch_size)
            buffer.buffer_img[buffer.current_index: buffer.current_index + offset].data.copy_(x[:offset])
            buffer.buffer_label[buffer.current_index: buffer.current_index + offset].data.copy_(y[:offset])


            buffer.current_index += offset
            buffer.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return list(range(buffer.current_index, buffer.current_index + offset))

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, buffer.n_seen_so_far).long()

        valid_indices = (indices < buffer.buffer_img.size(0)).long()


        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        ## zyq: choose the samples with least replay times to be replaced


        # store_sample_num = torch.sum(valid_indices.detach().cpu().numpy())
        # idx_buffer = torch.argsort(buffer.buffer_replay_times.detach().cpu().numpy())[:store_sample_num]
        # idx_buffer = idx_buffer.cuda()
        ############################

        buffer.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return []

        assert idx_buffer.max() < buffer.buffer_img.size(0)
        assert idx_buffer.max() < buffer.buffer_label.size(0)
        # assert idx_buffer.max() < self.buffer_task.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        idx_map = {idx_buffer[i].item(): idx_new_data[i].item() for i in range(idx_buffer.size(0))}
        ## zyq: save replay_times
        for i in list(idx_map.keys()):
            replay_times = buffer.buffer_replay_times[i].detach().cpu().numpy()
            buffer.unique_replay_list.append(int(replay_times))
            buffer.buffer_replay_times[i]=0
            buffer.buffer_last_replay[i]=0
            label = int(buffer.buffer_label[i].detach().cpu().numpy())
            buffer.replay_sample_label.append(label)
        # perform overwrite op
        buffer.buffer_img[list(idx_map.keys())] = x[list(idx_map.values())]
        buffer.buffer_label[list(idx_map.keys())] = y[list(idx_map.values())]
        return list(idx_map.keys())