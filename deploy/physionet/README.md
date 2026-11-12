# Running the full pipeline on AWS

Use this when the laptop can't hold the run (`make physionet-plan` says so) or shouldn't be tied up for it.

This stack is one private machine: encrypted disk, no inbound ports, an idle stop and a budget alarm. There's no S3 bucket and no cluster, because the pipeline is one sequential job. Data arrives over HTTPS from physionet.org. The MIMIC-IV page also offers a download with the AWS command-line tools, which may be faster from inside AWS, but the pipeline doesn't use that route yet.

**Check first:** the data and its Data Use Agreement stay yours. Don't share this machine's access with anyone, and the PhysioNet rule on model APIs applies here too (see [docs/PHYSIONET.md](../../docs/PHYSIONET.md)).

**Before applying, check two account limits.** Both stopped the first apply:

* **The account's plan.** An account on AWS's Free plan can only launch free-tier instance types, and the largest of those (`m7i-flex.large`, 2 vCPU and 8 GB) is too small for this run. The apply then fails with `InvalidParameterCombination: ... not eligible for Free Tier`. Upgrading the plan is a billing decision, made in the console (Billing → Free Tier → Upgrade plan). Remaining credits carry over and are spent first. `aws freetier get-account-plan-state --region us-east-1` shows the plan.
  To stay on the Free plan, apply with `-var instance_type=m7i-flex.large` (the largest free-tier type) and expect a lean plan: the planner scales its time estimates by thread count, so on 2 threads it gives up never-septic stays and stages to fit `HOURS`. The boot script adds a 24 GB swap file on the encrypted disk so building the grid can't run out of memory. The planner never counts swap: modelling is fitted into real RAM.
* **The region.** This account's organization policy allows only `us-east-2`, so that's the default here. Elsewhere, EC2 and SSM calls fail with an explicit deny.

```bash
cd deploy/physionet/terraform
terraform init
eval "$(aws configure export-credentials --format env)"   # Terraform can't read an `aws login` session itself
terraform apply -var alert_email=you@example.com
```

The Terraform state stays on your machine, git-ignored. It holds no password.

Store the login yourself. The `read -s` keeps the password off your screen and out of your shell history:

```bash
aws ssm put-parameter --region us-east-2 --name /physionet/username --type String --value <your-username>
```

```bash
read -rs PW && aws ssm put-parameter --region us-east-2 --name /physionet/password --type SecureString --value "$PW"; unset PW
```

Then connect (the `connect` output prints this) and run:

```bash
sudo /srv/aegis/Sepsis/deploy/physionet/run.sh doctor
```

```bash
sudo /srv/aegis/Sepsis/deploy/physionet/run.sh up
```

On this machine `run.sh` uses the `cloud` scale. It has two differences from the laptop's `full` scale, both on dev folds only:

* every never-septic stay if the planner can afford it, instead of 25%;
* when time runs short, it gives up negatives (keep rate 1.0 → 0.5 → 0.25) before it drops any stage.

It runs one booster thread per physical core (every vCPU on a small machine), and fetches each table over 4 byte ranges. The planner fits the run into `HOURS` (default 30: at most about $25 of compute on an m7i.4xlarge at ~$0.81 an hour, about $3 on the free tier). `SCALE=full`, `HOURS=…` or `PARALLEL=…` before the command overrides each.

Watch it with `sudo /srv/aegis/Sepsis/deploy/physionet/run.sh status`: the stages, then fits done of planned, memory and an ETA. A second run on the same machine plans with the speed the first one measured.

The job survives a disconnect only if you start it inside `tmux` or `nohup`. If the machine stops, start it again and rerun `up`: finished stages and fits are kept.

When you're finished, `terraform destroy` deletes the machine **and its disk**, which is the point. Copy out only the released report, never the lake.

The `doctor` check reports disk encryption as "not measurable here" inside a container; on this stack the `encrypted = true` above is what guarantees it.

## One heavy job at a time

The worker has 2 vCPUs and 7.6 GB of RAM. Two heavy jobs at once run it out of memory, and
several agents share it. So every heavy job takes one lock, and the kernel enforces it:

```bash
sudo install -m 755 deploy/physionet/heavy.sh /usr/local/sbin/aegis-heavy
sudo systemd-run --unit yaib-full aegis-heavy run yaib-full -- docker run --rm ...  # waits its turn
sudo aegis-heavy run probe --no-wait -- ...    # exits 75 at once if the box is busy
aegis-heavy status                             # "free", or who holds it and since when
```

The lock is `flock` on `/run/lock/aegis-heavy.lock`. It is released when its holder exits or
is killed, and a reboot clears it. Children inherit it, so a pipeline holds it across all its
stages. Keep `docker run` in the foreground (no `-d`), or the container outlives the lock.
A unit started before the lock existed is covered with `aegis-heavy hold-while <unit>`.

`sudo bash deploy/physionet/test_heavy.sh` checks it on a private lock file (11 checks; with
the `flock` call removed, 6 of them fail).

The lock covers one job at a time. A reservation that spans several jobs, with gaps between
them, is still agreed between the agents (on Axis, with a release marker).

`yaib_conductor.sh` uses such a gap on purpose. After a benchmark sweep it writes the
release marker, lets the Forge optimization worker run its one six-hour job, and only then
queues the benchmark's 25 runs behind the lock:

```bash
sudo systemd-run --unit aegis-conductor-<rev> --setenv SWEEP=<sweep rev> --setenv FINAL=<rev> \
    /srv/aegis/launch/yaib_conductor.sh
```

Name the unit anything but `yaib-*` or `eicu-*`: the Forge worker reads those names as the
benchmark still running and will not start.

### The shared checkout on the worker is pinned

`/srv/aegis/Sepsis` stays at commit `549804e`. The Forge optimization job records a
fingerprint of every `backend/ml/*.py` in that checkout when its private bundle is
prepared, and refuses to start when the fingerprint differs. An earlier benchmark
pipeline ran `git reset --hard` there on each launch, so on 2026-11-12 both of the Forge
job's attempts exited within five seconds, before any fit. The checkout was put back, the
job's own startup checks were dry-run in its image (all four pass), and the worker was
restarted with a clean state.

Never `reset`, `checkout` or `pull` in that directory. Deploy newer code by export, which
leaves the working tree alone:

```bash
git -C /srv/aegis/Sepsis fetch --depth 1 origin main
git -C /srv/aegis/Sepsis archive FETCH_HEAD backend/ml/yaib backend/ml/scripts | tar -x -C <staging dir>
```
