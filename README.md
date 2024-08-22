# python-chat-rag

Simple chat app integrating LangChain + RAG in a Python (Flask) server created for AI Series Part V blog post

## Running

`cp .env.example .env`

Edit .env and add your OpenAI API key

Via docker: `docker compose up`

Note: if you run using Docker, make sure to update the `destination` URL in the `next.config.mjs` as the following: 
```
destination: 'http://flask:5328/:path*',
```

This will make sure to use the Docker container's name instead of localhost

Or you can run the apps separately:

 `pnpm run dev`

 And in another terminal:

 `cd flask && flask run --host=0.0.0.0 --debug -p 5328`

Either way, the app will start at `localhost:3000`
