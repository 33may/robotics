---
name: Remote Robot Machine SSH Fingerprint
description: Canonical ED25519 host-key fingerprint of the 5090 remote robot machine, used to identify it across DHCP IP changes
type: reference
originSessionId: 5831fc1f-f3a8-4242-8572-532eea191b91
---
## Canonical fingerprint

ED25519 host key SHA256: **`pPDbgPe7yJwJJjPPE6Gbv+MokF7qZ379z1EcG4AfJD0`**

Confirmed identical across two past IPs:
- `10.11.100.151` (before 2026-05-11)
- `10.11.101.240` (2026-05-11 onwards, per remote_machine.md)

## How to find the machine when DHCP moves it

```bash
# 1. ARP-scan the /23 (need sudo, requires arp-scan installed)
sudo arp-scan --interface=wlp3s0 10.11.100.0/23 | tee /tmp/arp.txt
# 2. Filter SSH-open hosts
awk '/^10\./{print $1}' /tmp/arp.txt | sort -u > /tmp/live.txt
nmap -p 22 --open -T4 -iL /tmp/live.txt | awk '/Nmap scan report/{print $5}' > /tmp/ssh.txt
# 3. Compare each live SSH host's ED25519 fingerprint to the canonical one
for ip in $(cat /tmp/ssh.txt); do
  fp=$(ssh-keyscan -T 3 -t ed25519 "$ip" 2>/dev/null | ssh-keygen -lf - 2>/dev/null | awk '{print $2}')
  [ "$fp" = "SHA256:pPDbgPe7yJwJJjPPE6Gbv+MokF7qZ379z1EcG4AfJD0" ] && echo "ROBOT @ $ip"
done
```

If no host matches, the robot is off or off the subnet — not "wrong IP."

## Username & password (for reference)

- User: `vbti`
- Password: `vbti25robot`

(Network details, hostname, paths — see `remote_machine.md`)
