"""Monitor training progress and check stop conditions."""
import re
import sys
import time

LOG_FILE = "/home/ciallo/claude/eet/train_run5.log"
LOG3 = 1.0986
STALE_THRESHOLD = 600  # seconds without log update = probably dead

def parse_log():
    try:
        data = open(LOG_FILE, 'rb').read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return [], [], False, False
    data = data.replace('\r', '\n')
    data = re.sub(r'\x1b\[[0-9;]*m', '', data)

    done_lines = re.findall(r'Epoch\s+\d+ done.*', data)
    val_lines = re.findall(r'Validation avg reward.*', data)
    estop = 'Early stopping' in data
    nan = any('nan' in l.lower() for l in data.split('\n'))
    return done_lines, val_lines, estop, nan


def parse_epoch_metrics(line):
    """Extract metrics from an epoch done line."""
    m = {}
    ep = re.search(r'Epoch\s+(\d+) done', line)
    if ep: m['epoch'] = int(ep.group(1))

    rm = re.search(r'AvgR/ep\s+([-\d.]+)', line)
    if rm: m['avg_r_ep'] = float(rm.group(1))

    pm = re.search(r'policy_loss=([-\d.e+]+)', line)
    if pm: m['policy_loss'] = float(pm.group(1))

    vm = re.search(r'value_loss=([-\d.e+]+)', line)
    if vm: m['value_loss'] = float(vm.group(1))

    em = re.search(r'entropy=([-\d.]+)', line)
    if em: m['entropy'] = float(em.group(1))

    am = re.search(r'adv\[μ=([-\d.]+)\s+σ=([-\d.]+)\s+\+%=(\d+\.\d+)\]', line)
    if am:
        m['adv_mean'] = float(am.group(1))
        m['adv_std'] = float(am.group(2))
        m['adv_frac_pos'] = float(am.group(3))

    return m


def check_stop_conditions(metrics_list):
    """Check all stop conditions, return (should_stop, reason)."""
    if not metrics_list:
        return False, ""

    # 1. Entropy stuck at log(3) for 15+ epochs
    recent_entropy = [m['entropy'] for m in metrics_list[-15:] if 'entropy' in m]
    if len(recent_entropy) >= 15:
        if all(abs(e - LOG3) < 0.01 for e in recent_entropy):
            return True, f"STOP: Entropy stuck at log(3)={LOG3:.3f} for {len(recent_entropy)} epochs. Last: {recent_entropy[-1]:.4f}"

    # 3. Value loss spikes >200 repeatedly (3+ in last 10 epochs)
    recent_vl = [m['value_loss'] for m in metrics_list[-10:] if 'value_loss' in m]
    spikes = sum(1 for v in recent_vl if v > 200)
    if spikes >= 3:
        return True, f"STOP: Value loss >200 in {spikes} of last {len(recent_vl)} epochs. Values: {recent_vl}"

    # 4. NaN
    for m in metrics_list:
        for k in ['policy_loss', 'value_loss', 'entropy']:
            if k in m and (str(m[k]).lower() == 'nan' or str(m[k]).lower() == 'inf'):
                return True, f"STOP: NaN/Inf in {k} at epoch {m.get('epoch', '?')}"

    # 5. adv_mean < -10 sustained for 10+ epochs
    recent_adv = [m['adv_mean'] for m in metrics_list[-10:] if 'adv_mean' in m]
    if len(recent_adv) >= 10 and all(a < -10 for a in recent_adv):
        return True, f"STOP: adv_mean < -10 for {len(recent_adv)} epochs. Values: {recent_adv}"

    return False, ""


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    prev_count = 0

    while True:
        done_lines, val_lines, estop, nan = parse_log()
        metrics = [parse_epoch_metrics(l) for l in done_lines]
        metrics = [m for m in metrics if m]

        if mode == "watch":
            if len(metrics) > prev_count:
                new_epochs = metrics[prev_count:]
                for m in new_epochs:
                    e = m.get('epoch', '?')
                    print(f"\n=== Epoch {e} Summary ===")
                    for k, v in m.items():
                        if k != 'epoch':
                            print(f"  {k}: {v}")
                prev_count = len(metrics)
        else:
            # Print full summary
            print(f"=== Training Status: {len(metrics)} epochs completed, {len(val_lines)} validations ===\n")

            if len(metrics) >= 10:
                # Print every 10th epoch
                for m in metrics:
                    e = m.get('epoch', 0)
                    if e % 10 == 0 or e == 1 or e == len(metrics):
                        parts = [f"Ep{e:>3d}"]
                        if 'avg_r_ep' in m: parts.append(f"R/ep={m['avg_r_ep']:.1f}")
                        if 'entropy' in m: parts.append(f"H={m['entropy']:.4f}")
                        if 'policy_loss' in m: parts.append(f"πL={m['policy_loss']:.4e}")
                        if 'value_loss' in m: parts.append(f"VL={m['value_loss']:.1f}")
                        if 'adv_mean' in m: parts.append(f"advμ={m['adv_mean']:.3f}")
                        if 'adv_std' in m: parts.append(f"advσ={m['adv_std']:.1f}")
                        print("  " + " | ".join(parts))
            else:
                for m in metrics:
                    e = m.get('epoch', 0)
                    parts = [f"Ep{e:>3d}"]
                    if 'avg_r_ep' in m: parts.append(f"R/ep={m['avg_r_ep']:.1f}")
                    if 'entropy' in m: parts.append(f"H={m['entropy']:.4f}")
                    if 'policy_loss' in m: parts.append(f"πL={m['policy_loss']:.4e}")
                    if 'value_loss' in m: parts.append(f"VL={m['value_loss']:.1f}")
                    if 'adv_mean' in m: parts.append(f"advμ={m['adv_mean']:.3f}")
                    print("  " + " | ".join(parts))

            print(f"\nValidation results:")
            for vl in val_lines:
                print(f"  {vl[:150]}")

            should_stop, reason = check_stop_conditions(metrics)
            if should_stop:
                print(f"\n*** {reason} ***")
            else:
                print(f"\nNo stop conditions triggered.")

            if estop:
                print("EARLY STOPPING triggered by training loop.")
            if nan:
                print("NaN detected in log!")

        if mode != "watch":
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
