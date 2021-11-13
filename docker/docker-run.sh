#!/bin/bash
script_dir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
function git_cmd() {
  cd $script_dir && git $@;
}

docker_tag=latest
cwd=$(pwd -P)

print_help() {
  echo "USAGE: $0 [options] cmd";
  echo "  Available options are:";
  echo "";
  echo "   -i|--interactive: ";
  echo "           start an interactive docker session";
  echo "   -s|--sudo: ";
  echo "           use root account and run in priviledged mode";
  echo "   -t|--image: ";
  echo "           specify a docker image to run, will skip image build";
  echo "   -h|--help: ";
  echo "           print this message";
  echo "";
  echo "";
}

options=
# additional options
while [[ $# -gt 0 ]]; do
  key="$1"
  if [[ $key != -* ]]; then break; fi
  case $key in
  -i|--interactive)
    echo "INFO: starting interactive docker session"
    options="$options -it"
    ;;
  -s|--sudo)
    echo "INFO: using root account in docker"
    use_root=1
    ;;
  -t|--image)
    image=$2
    if [ -z "image" ]; then
      echo "ERROR: empty image"
      exit 2
    else
      echo "INFO: using image $image"
    fi
    shift
    ;;
  *)
    # unknown option
    echo "ERROR: Failed to recongize argument '$1'"
    print_help
    exit 1
    ;;
  esac
  shift # past argument or value
done

if [ "$#" -lt 1 ]; then
  print_help
  exit 1
fi

if [ -z "$image" ]; then
  image="merlin-dse:$docker_tag"
fi

if [ -z "$use_root" ]; then
  options="$options -u $(id -u):$(id -g)"
else
  options="$options --privileged"
fi

# find mountpoint for pwd
function get_base {
  local dir="$1";
  while [ ! $(dirname $dir) = "/" ]; do
    dir=$(dirname $dir);
    if mountpoint $dir &> /dev/null; then
      break;
    fi;
  done;
  echo $dir;
}

declare -A illegal_base
illegal_base["/bin"]=1
illegal_base["/dev"]=1
illegal_base["/etc"]=1
illegal_base["/lib"]=1
illegal_base["/lib64"]=1
illegal_base["/media"]=1
illegal_base["/mnt"]=1
illegal_base["/opt"]=1
illegal_base["/proc"]=1
illegal_base["/root"]=1
illegal_base["/run"]=1
illegal_base["/sbin"]=1
illegal_base["/srv"]=1
illegal_base["/sys"]=1
illegal_base["/usr"]=1
illegal_base["/var"]=1

function legal_base {
  local dir="$1";
  while [ ! $(dirname $dir) = "/" ]; do
    tmp_dir=$(dirname $dir)
    if [ ${illegal_base[$tmp_dir]} ]; then
      break;
    fi;
    dir=$tmp_dir;
    if mountpoint $dir &> /dev/null; then
      break;
    fi;
  done;
  if [ ${illegal_base[$dir]} ]; then
    echo "Error: $dir illegal path"
    exit 1
  fi
  echo $dir;
}

declare -A mounted_base

# handle vendor tools
function add_env {
  local var=$1;
  local val="${!var}";
  if [ -z "$val" -o ! -d "$val" ]; then
    echo "Error: $val env variable is invalid"
    exit 1
  fi
  baseval=$(legal_base $val)
  if [ ! ${mounted_base[$baseval]} ]; then
    options="$options -v $baseval:$baseval"
    mounted_base[$baseval]=1
  fi
  options="$options -e "$var=$val""
}

#add_env XILINX_SDX
add_env XILINX_VITIS
add_env XILINX_XRT
add_env XILINX_VIVADO
#if [ -z "$XILINX_SDK" ]; then
#  #XILINX_SDK=$(dirname $(dirname $XILINX_SDX))/SDK/$(basename $XILINX_SDX)
#  XILINX_SDK=$(dirname $(dirname $XILINX_VITIS))/SDK/$(basename $XILINX_VITIS)
#fi


function add_license {
  local var=$1;
  local val="${!var}";
  if [ -z "$val" ]; then
    return
  fi
  local vals=$(echo "$val" | tr ":" "\n")
  local realval=""
  for vv in $vals; do
    if [ ! -f $vv ]; then
      realval=$realval:$vv
      continue
    fi
    local realvv=$(realpath $vv)
    realval=$realval:$realvv
    if [ -f "$realvv" ]; then
      local baseval=$(legal_base $realvv)
      if [ ! ${mounted_base[$baseval]} ]; then
        options="$options -v $baseval:$baseval"
        mounted_base[$baseval]=1
      fi
    fi
  done
  options="$options -e "$var=$realval""
}

add_license LM_LICENSE_FILE
add_license FALCONLM_LICENSE_FILE
add_license XILINXD_LICENSE_FILE

#options="$options -v $script_dir/../license:/opt/merlin/license"
#options="$options -e LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/merlin/license/license.lic:$cwd/license.lic"
#options="$options -e FALCONLM_LICENSE_FILE=$FALCONLM_LICENSE_FILE"
#options="$options -e XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"
options="$options -e MERLIN_AUTO_DEVICE_XILINX=$MERLIN_AUTO_DEVICE_XILINX"
options="$options -e MERLIN_AUTO_DEVICE_INTEL=$MERLIN_AUTO_DEVICE_INTEL"

base_dir=$(get_base $cwd)
if [ ${illegal_base[$base_dir]+abc} ]; then
  echo "[merlin-cmd] ERROR: running from $cwd is not supported, please run from a different directory"
  exit 1
fi

# Xilinx tool license needs
options="$options --net host"

echo " \
docker run \
    $options \
    -v "$base_dir":"$base_dir" \
    -w="$cwd" \
    -e "WITH_DOCKER=1" \
    --rm \
    -t \
    $image \
    $@ "

docker run \
    $options \
    -v "$base_dir":"$base_dir" \
    -w="$cwd" \
    -e "WITH_DOCKER=1" \
    --rm \
    -t \
    $image \
    $@

