import jwt
from datetime import datetime, timedelta
import argparse
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN_SECRET_KEY = os.getenv('TOKEN_SECRET_KEY')
if not TOKEN_SECRET_KEY:
    raise ValueError("TOKEN_SECRET_KEY environment variable is required")


def generate_token(secret_key: str, app: str, days: int = 1) -> str:
    expire = (datetime.now() + timedelta(days=days)).isoformat()
    return jwt.encode({'expire': expire, 'app': app}, secret_key, algorithm='HS256')


def _parse_args():
    parser = argparse.ArgumentParser(description='Generate JWT token')
    parser.add_argument('app', help='Application name')
    parser.add_argument('--expire', type=int, default=1, help='Expiration in days')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    token = generate_token(TOKEN_SECRET_KEY, args.app, args.expire)
    print(f"Your token is:\n{token}")
