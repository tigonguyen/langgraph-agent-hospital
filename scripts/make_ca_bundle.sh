#!/bin/sh
# Build certs/ca-bundle.pem = certifi's roots + this machine's trusted roots.
#
# Needed when HTTPS is TLS-intercepted (here: Cloudflare Gateway / Zero Trust). The
# intercepting root lives in the macOS keychain, so curl trusts it but Python — which uses
# certifi's own bundle — does not, and every API call fails as the Anthropic SDK's opaque
# "APIConnectionError: Connection error."  `.env` points SSL_CERT_FILE at the output.
set -e
cd "$(dirname "$0")/.."
mkdir -p certs
{
  .venv/bin/python -c 'import certifi; print(open(certifi.where()).read())'
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain
  security find-certificate -a -p /Library/Keychains/System.keychain
} > certs/ca-bundle.pem
echo "certs/ca-bundle.pem: $(grep -c 'BEGIN CERTIFICATE' certs/ca-bundle.pem) certificates"
