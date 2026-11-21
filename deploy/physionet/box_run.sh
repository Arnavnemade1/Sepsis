#!/bin/bash
# Run a local shell script on the PhysioNet worker over SSM and print its result.
#
#   deploy/physionet/box_run.sh status.sh [timeout-seconds]
#
# Fails loudly instead of hanging: an expired `aws login` session (renew with `! aws login`
# in Claude Code, or `aws login` in a terminal; a background `aws login` never gets the
# browser callback), a stopped instance (`aws ec2 start-instances`), or a command still
# running at the deadline (exit 124; the box keeps running it). Exit = the script's exit.
# Print aggregates only: never rows (see docs/PHYSIONET.md).
set -uo pipefail
script=${1:?usage: box_run.sh <script.sh> [timeout-seconds]}
deadline=${2:-600}
BOX=${AEGIS_BOX:-i-02cda9b7d29074588}
export AWS_REGION=${AWS_REGION:-us-east-2}

aws sts get-caller-identity >/dev/null 2>&1 || { echo "box_run: AWS credentials expired or missing: run 'aws login'" >&2; exit 3; }
state=$(aws ec2 describe-instances --instance-ids "$BOX" --query 'Reservations[0].Instances[0].State.Name' --output text)
[ "$state" = running ] || { echo "box_run: $BOX is $state (start it: aws ec2 start-instances --instance-ids $BOX)" >&2; exit 4; }

cmd=$(base64 < "$script" | tr -d '\n')
id=$(aws ssm send-command --instance-ids "$BOX" --document-name AWS-RunShellScript \
  --timeout-seconds 60 --parameters "executionTimeout=$deadline,commands=[\"echo $cmd | base64 -d | bash\"]" \
  --query Command.CommandId --output text) || { echo "box_run: send-command failed" >&2; exit 5; }

end=$(( $(date +%s) + deadline + 30 ))
while :; do
  st=$(aws ssm get-command-invocation --command-id "$id" --instance-id "$BOX" --query Status --output text 2>/dev/null) \
    || { aws sts get-caller-identity >/dev/null 2>&1 || { echo "box_run: credentials expired while waiting ($id)" >&2; exit 3; }; st=Pending; }
  case "$st" in Success|Failed|Cancelled|TimedOut) break ;; esac
  [ "$(date +%s)" -lt "$end" ] || { echo "box_run: still $st after ${deadline}s ($id)" >&2; exit 124; }
  sleep 5
done
aws ssm get-command-invocation --command-id "$id" --instance-id "$BOX" \
  --query '[StandardOutputContent,StandardErrorContent,ResponseCode]' --output json \
  | python3 -c 'import json,sys; o,e,rc=json.load(sys.stdin); sys.stdout.write(o); sys.stderr.write(e); sys.exit(rc if rc >= 0 else 1)'
