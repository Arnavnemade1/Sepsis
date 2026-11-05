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
