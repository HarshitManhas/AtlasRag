FROM golang:1.24 AS build
RUN CGO_ENABLED=0 go install github.com/minio/minio@7aac2a2c5b7c882e68c1ce017d8256be2feea27f

FROM alpine:3.22
RUN apk add --no-cache ca-certificates curl
COPY --from=build /go/bin/minio /usr/local/bin/minio
ENTRYPOINT ["minio"]
