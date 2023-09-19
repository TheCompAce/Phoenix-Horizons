openssl genrsa -out certs\server.key 2048

REM Set to your path
set OPENSSL_CONF=C:\Program Files\OpenSSL-Win64\bin\cnf\openssl.cnf

openssl req -new -x509 -key certs\server.key -out certs\server.crt -days 3650