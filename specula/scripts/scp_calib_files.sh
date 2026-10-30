#!/usr/bin/env bash
# Copy the files listed by collect_calib_files.py (one absolute path per line)
# to another machine with scp, keeping their paths.
#
#   scp_calib_files.sh [-n] [-s SSH_HOST] [-b BASE] [-m LOCAL_PREFIX=REMOTE_PREFIX]... LIST SCP_HOST
#
#   SCP_HOST  where the files are sent with scp (user@host)
#   -s   host where the remote directories are created with ssh, when it isn't
#        SCP_HOST (e.g. a login node, while SCP_HOST is a data-transfer node
#        that only accepts scp). Both must see the same filesystem.
#   -n   dry run: print what would be done
#   -v   verbose ssh/scp (-v), to debug connection or authentication problems
#   -b   remote base directory, prepended to every path, e.g. with -b /data/copy
#          /raid1/guido/PASSATA/LBT/ifunc/x.fits -> /data/copy/raid1/guido/PASSATA/LBT/ifunc/x.fits
#   -m   remap a path prefix on the remote side (repeatable, first match wins), e.g.
#          -m /home/frossi/dev/SPECULA=/home/rossi/SPECULA -m /raid1/guido=/data/guido
#        Paths that match no prefix keep the same path on the remote machine.
#        -m is applied first, then -b.
#
# All the remote directories are created first with one ssh (mkdir -p), then
# each directory's files are sent with one scp (-p keeps the timestamps).
# The connection to each host is shared by all its commands, so a password
# is asked at most once per host. At the end, the remote sizes are checked
# against the local ones.
set -euo pipefail

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

dry_run=0
verbose=()
base=
ssh_host=
maps=()
while getopts "nvs:b:m:h" opt; do
    case $opt in
        n) dry_run=1 ;;
        v) verbose=(-v) ;;
        s) ssh_host=$OPTARG ;;
        b) base=${OPTARG%/} ;;
        m) [[ $OPTARG == *=* ]] || { echo "-m needs LOCAL=REMOTE" >&2; exit 1; }
           maps+=("$OPTARG") ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 2 ]] || usage
list=$1
host=$2
ssh_host=${ssh_host:-$host}

remote_path() {
    local p=$1 m
    for m in "${maps[@]+"${maps[@]}"}"; do
        local from=${m%%=*} to=${m#*=}
        if [[ $p == "$from" || $p == "$from"/* ]]; then
            p="$to${p#"$from"}"
            break
        fi
    done
    echo "$base$p"
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

# remote directory -> local files going there; remote file -> local size
declare -A by_dir size_of
n_files=0
total=0
while IFS= read -r f || [[ -n $f ]]; do
    [[ -z $f || $f == \#* ]] && continue
    if [[ ! -f $f ]]; then
        log "not found locally, skipped: $f"
        continue
    fi
    dest=$(remote_path "$f")
    by_dir[$(dirname "$dest")]+="$f"$'\n'
    size_of[$dest]=$(stat -c %s "$f")
    total=$((total + size_of[$dest]))
    n_files=$((n_files + 1))
done < "$list"
[[ $n_files -gt 0 ]] || { log "nothing to copy"; exit 1; }
mapfile -t dirs < <(printf '%s\n' "${!by_dir[@]}" | sort)

log "$n_files files, $(human $total), into ${#dirs[@]} directories"
log "  scp to:            $host"
log "  directories made on: $ssh_host"
[[ -n $base ]] && log "  remote base:       $base"

ctl=$(mktemp -u "${TMPDIR:-/tmp}/scp_calib.XXXXXX")
ssh_opts=(-o ControlMaster=auto -o ControlPath="$ctl-%C" -o ControlPersist=60 "${verbose[@]+"${verbose[@]}"}")

run() {
    if [[ $dry_run -eq 1 ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

# 1. scp doesn't create directories: make them all first, on ssh_host.
#    This goes through a remote shell, so the paths are quoted.
log "[1/3] creating ${#dirs[@]} directories on $ssh_host:"
mkdir_cmd="mkdir -p"
for d in "${dirs[@]}"; do
    log "        $d"
    mkdir_cmd+=" '$d'"
done
start=$SECONDS
if run ssh "${ssh_opts[@]}" "$ssh_host" "$mkdir_cmd"; then
    log "      done ($((SECONDS - start)) s)"
else
    rc=$?
    log "      FAILED (ssh exit code $rc): cannot create the directories on $ssh_host"
    log "      (can you log in there? is ${base:-/} writable?)"
    exit 1
fi

# 2. One scp per directory. The remote path is NOT quoted: recent OpenSSH scp
#    (SFTP protocol) takes it literally, quotes included.
log "[2/3] copying to $host"
failed=0
k=0
for d in "${dirs[@]}"; do
    k=$((k + 1))
    mapfile -t files < <(printf '%s' "${by_dir[$d]}")
    [[ $d == *[[:space:]]* ]] && log "      WARNING: $d contains spaces, scp may split it"
    bytes=0
    for f in "${files[@]}"; do bytes=$((bytes + $(stat -c %s "$f"))); done
    log "  ($k/${#dirs[@]}) ${#files[@]} file(s), $(human $bytes) -> $host:$d/"
    for f in "${files[@]}"; do log "        $(basename "$f")"; done
    start=$SECONDS
    if run scp -p "${ssh_opts[@]}" "${files[@]}" "$host:$d/"; then
        log "        ok ($((SECONDS - start)) s)"
    else
        rc=$?
        log "        FAILED (scp exit code $rc)"
        failed=$((failed + 1))
    fi
done

# 3. Check what arrived: remote size of every file, compared with the local one
if [[ $dry_run -eq 0 ]]; then
    log "[3/3] checking the remote files on $ssh_host"
    stat_cmd="stat -c '%s %n'"
    for dest in "${!size_of[@]}"; do stat_cmd+=" '$dest'"; done
    declare -A remote_size
    while read -r sz name; do
        remote_size[$name]=$sz
    done < <(ssh "${ssh_opts[@]}" "$ssh_host" "$stat_cmd" 2>/dev/null || true)
    bad=0
    for dest in $(printf '%s\n' "${!size_of[@]}" | sort); do
        if [[ -z ${remote_size[$dest]:-} ]]; then
            log "      MISSING    $dest"; bad=$((bad + 1))
        elif [[ ${remote_size[$dest]} != "${size_of[$dest]}" ]]; then
            log "      WRONG SIZE $dest (${remote_size[$dest]} instead of ${size_of[$dest]})"; bad=$((bad + 1))
        fi
    done
    for h in "$ssh_host" "$host"; do
        ssh -o ControlPath="$ctl-%C" -O exit "$h" 2>/dev/null || true
    done
    if [[ $bad -eq 0 && $failed -eq 0 ]]; then
        log "all $n_files files arrived ($(human $total))"
    else
        log "$bad of $n_files files missing or incomplete, $failed scp command(s) failed"
        exit 1
    fi
fi
