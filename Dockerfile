ARG PNPM_VERSION=8.7.1
FROM node:20.6.1

COPY . ./python-chat-rag
WORKDIR /python-chat-rag

RUN npm install -g pnpm@${PNPM_VERSION}

ENTRYPOINT pnpm install && pnpm run build && pnpm start