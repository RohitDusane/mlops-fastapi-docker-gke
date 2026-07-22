#!/bin/bash

dnf install -y docker git

systemctl enable docker
systemctl start docker

curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 0644

