script_dir=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)

docker image build \
    -t merlin-dse-test \
    -f $script_dir/Dockerfile \
    $script_dir/..

