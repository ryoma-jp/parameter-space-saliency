#! /bin/bash

# Create .env file with UID and GID
cat > .env << EOF
UID=$(id -u)
GID=$(id -g)
EOF

echo ".env file created successfully"
