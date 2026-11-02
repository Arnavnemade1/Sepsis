# Running the full pipeline on AWS

Use this when the laptop can't hold the run (`make physionet-plan` says so) or shouldn't be tied up for it.

This stack is one private machine: encrypted disk, no inbound ports, an idle stop and a budget alarm. There's no S3 bucket and no cluster, because the pipeline is one sequential job. Data arrives over HTTPS from physionet.org: I found no MIMIC-IV bucket on AWS, and the MIT-LCP cloud docs list BigQuery and Google Cloud Storage for it.

**Check first:** the data and its Data Use Agreement stay yours. Don't share this machine's access with anyone, and the PhysioNet rule on model APIs applies here too (see [docs/PHYSIONET.md](../../docs/PHYSIONET.md)).

```bash
cd deploy/physionet/terraform
terraform init
terraform apply -var alert_email=you@example.com
```

Store the login yourself. The `read -s` keeps the password off your screen and out of your shell history:

```bash
aws ssm put-parameter --name /physionet/username --type String --value <your-username>
```

```bash
read -rs PW && aws ssm put-parameter --name /physionet/password --type SecureString --value "$PW"; unset PW
```

Then connect (the `connect` output prints this) and run:

```bash
sudo /srv/aegis/Sepsis/deploy/physionet/run.sh doctor
```

```bash
sudo /srv/aegis/Sepsis/deploy/physionet/run.sh up
```

The job survives a disconnect only if you start it inside `tmux` or `nohup`. If the machine stops, start it again and rerun `up`: finished stages and fits are kept.

When you're finished, `terraform destroy` deletes the machine **and its disk**, which is the point. Copy out only the released report, never the lake.

The `doctor` check reports disk encryption as "not measurable here" inside a container; on this stack the `encrypted = true` above is what guarantees it.
