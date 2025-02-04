# Jupyter Lab configuration
c = get_config()  #noqa

# To set up password authentication do these on the host machine:
#
# 1. Install passlib:
#    pip install passlib
#
# 2. Generate a hashed password:
#    from passlib.hash import argon2
#    password = "your-password"
#    print(argon2.hash(password))
#
# 3. Copy the hash and paste it below:
c.PasswordIdentityProvider.hashed_password = ''  # Paste hash here

# this is needed for the jupyter lab to work
c.IdentityProvider.token = ''