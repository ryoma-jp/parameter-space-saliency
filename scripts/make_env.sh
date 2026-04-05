#! /bin/bash

set -eu

# Create .env file with UID, GID and USERNAME
USERNAME=${USER:-pss}

cat > .env << EOF
UID=$(id -u)
GID=$(id -g)
USERNAME=${USERNAME}
EOF

echo ".env file created successfully"
