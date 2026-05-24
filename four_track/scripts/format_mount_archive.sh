#!/usr/bin/env bash
# format_mount_archive.sh
#
# *** HISTORICAL — EXECUTED 2026-05-08 23:05 BY HAND, DO NOT RE-RUN ***
#
# Kept as a record of intent. The actual reformat used different
# parameters than this script:
#   - GPT + single partition /dev/sda1   (this script: whole disk, no PT)
#   - mkfs.ext4 -m 0 -T largefile4       (this script: -m 1)
#   - label "MachineLearning"            (this script: birdclef-archive)
#   - mount at /mnt/MachineLearning      (this script: /mnt/archive)
# UUID of the live filesystem: 70aa93f5-d422-4e81-8eb8-f5433cc064a5.
# See new_plan.md §14.21.7 step 4c for the executed plan.
#
# Original purpose (preserved verbatim below): format /dev/sda as ext4
# and mount at /mnt/archive for cold-archive use, after badblocks
# reports clean. Refuses to run if badblocks output is non-empty or
# missing. DESTRUCTIVE — wipes /dev/sda's partition table and existing
# exfat filesystem.

set -euo pipefail

DEVICE="/dev/sda"
MOUNTPOINT="/mnt/archive"
LABEL="birdclef-archive"
BADBLOCKS_OUT="/home/swatson/work/kaggle/BirdCLEF/four_track/log/badblocks_sda_20260505_131456.txt"
BADBLOCKS_LOG="/home/swatson/work/kaggle/BirdCLEF/four_track/log/post_xc_badblocks_20260505_131456.log"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

bold "=== Preflight checks ==="

# 1. badblocks must have finished (no running scan) AND output file must be empty
if pgrep -af "badblocks .* ${DEVICE}" >/dev/null; then
    red "ABORT: badblocks is still running on ${DEVICE}."
    pgrep -af "badblocks"
    exit 1
fi
green "[ok] no badblocks process running on ${DEVICE}"

if [[ ! -f "${BADBLOCKS_OUT}" ]]; then
    red "ABORT: badblocks output file missing: ${BADBLOCKS_OUT}"
    exit 1
fi
if [[ -s "${BADBLOCKS_OUT}" ]]; then
    red "ABORT: badblocks reported bad sectors:"
    head -20 "${BADBLOCKS_OUT}"
    exit 1
fi
green "[ok] badblocks output is empty (no bad sectors)"

# 2. Confirm the wrapper log shows a completed scan
if ! grep -q "Pass completed" "${BADBLOCKS_LOG}" 2>/dev/null && \
   ! grep -q "100\.00% done" "${BADBLOCKS_LOG}" 2>/dev/null; then
    yellow "WARN: badblocks log does not contain 'Pass completed' or '100.00% done'"
    yellow "      log: ${BADBLOCKS_LOG}"
    yellow "      proceeding only if you've manually verified the scan finished"
fi

# 3. Device must not be mounted anywhere
if mount | grep -E "^${DEVICE}[0-9]* " ; then
    red "ABORT: ${DEVICE} (or a partition) is mounted. Unmount first."
    exit 1
fi
green "[ok] ${DEVICE} is not mounted"

# 4. Mountpoint must exist and be empty
if [[ ! -d "${MOUNTPOINT}" ]]; then
    red "ABORT: mountpoint ${MOUNTPOINT} does not exist"
    exit 1
fi
if [[ -n "$(ls -A "${MOUNTPOINT}" 2>/dev/null)" ]]; then
    red "ABORT: mountpoint ${MOUNTPOINT} is not empty"
    ls -la "${MOUNTPOINT}"
    exit 1
fi
green "[ok] mountpoint ${MOUNTPOINT} exists and is empty"

# 5. Show the device map and demand confirmation
echo ""
bold "=== Current block device map ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,UUID,MOUNTPOINTS
echo ""
red "ABOUT TO WIPE ${DEVICE} (7.3 TB Seagate One Touch, currently exfat)"
red "The system disk is /dev/nvme0n1 — make sure ${DEVICE} is the EXTERNAL drive."
echo ""
read -r -p "Type 'yes' to proceed: " confirm
if [[ "${confirm}" != "yes" ]]; then
    yellow "Aborted by user."
    exit 0
fi

bold "=== 1/5: wipefs ${DEVICE} ==="
sudo wipefs -a "${DEVICE}"

bold "=== 2/5: mkfs.ext4 ${DEVICE} ==="
# -m 1: claw back ~290 GB from the default 5% root reserve (cold archive
#       use, root doesn't need emergency space here).
# -L:   filesystem label, useful for `mount LABEL=...` lookups.
# -F:   force mkfs on a whole-disk device that previously had a
#       partition table.
sudo mkfs.ext4 -F -m 1 -L "${LABEL}" "${DEVICE}"

bold "=== 3/5: capture UUID ==="
UUID="$(sudo blkid -s UUID -o value "${DEVICE}")"
echo "UUID=${UUID}"
if [[ -z "${UUID}" ]]; then
    red "ABORT: could not read UUID from ${DEVICE}"
    exit 1
fi

bold "=== 4/5: append fstab entry ==="
FSTAB_LINE="UUID=${UUID}  ${MOUNTPOINT}  ext4  defaults,nofail,x-systemd.automount  0  2"
echo "fstab line:"
echo "  ${FSTAB_LINE}"
if grep -q "${MOUNTPOINT}" /etc/fstab; then
    yellow "WARN: /etc/fstab already has an entry for ${MOUNTPOINT}; not modifying."
else
    sudo cp /etc/fstab /etc/fstab.backup-$(date +%Y%m%d-%H%M%S)
    echo "${FSTAB_LINE}" | sudo tee -a /etc/fstab >/dev/null
    sudo systemctl daemon-reload
fi

bold "=== 5/5: mount + chown ==="
sudo mount "${MOUNTPOINT}"
sudo chown swatson:swatson "${MOUNTPOINT}"

echo ""
bold "=== Done ==="
df -h "${MOUNTPOINT}"
echo ""
mount | grep "${MOUNTPOINT}"
