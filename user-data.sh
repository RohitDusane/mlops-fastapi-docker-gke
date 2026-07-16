#!/bin/bash
dnf install -y docker
systemctl enable --now docker
curl -sfL https://get.k3s.io | sh -
