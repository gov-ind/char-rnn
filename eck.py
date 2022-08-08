from pdb import set_trace

def eck():
    mp = { 0: 0 }
    out = [0]
    idx = 1

    while len(out) < 100:
        last = out[-1]
        if last in mp:
            out.append(idx - mp[last])
        else:
            out.append(0)
        mp[last] = idx
        idx += 1

    return out

print(eck())
