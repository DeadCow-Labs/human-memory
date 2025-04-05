from setuptools import setup, find_packages

setup(
    name="human-memory",
    version="0.1.0",
    author="Lautaro Suarez",
    author_email="lautaro.suarez.dev@gmail.com",
    description="A Python SDK for storing structured memory objects in Supabase using OpenAI's API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deadcow-labs/human-memory",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "supabase>=2.0.0",
        "python-dotenv>=1.0.0",
        "psycopg2-binary>=2.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 