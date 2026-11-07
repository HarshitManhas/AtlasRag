"""Store original documents in the configured S3/MinIO bucket."""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from atlas_rag.config import get_settings


def storage_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(connect_timeout=3, read_timeout=10, retries={"max_attempts": 1}),
    )


def store_document(key: str, content: bytes, content_type: str) -> None:
    bucket = get_settings().s3_bucket
    with storage_client() as client:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
                raise
            try:
                client.create_bucket(Bucket=bucket)
            except ClientError as create_error:
                if create_error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise
        client.put_object(Bucket=bucket, Key=key, Body=content, ContentType=content_type)


def load_document(key: str) -> bytes:
    with storage_client() as client:
        response = client.get_object(Bucket=get_settings().s3_bucket, Key=key)
        try:
            return response["Body"].read()
        finally:
            response["Body"].close()


def remove_document(key: str) -> None:
    with storage_client() as client:
        client.delete_object(Bucket=get_settings().s3_bucket, Key=key)
