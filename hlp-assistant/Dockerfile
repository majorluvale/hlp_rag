FROM node:24-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
ENV NODE_ENV=production PORT=3000 DATA_DIR=/app/data DOCS_DIRS=/app/data/docs
VOLUME ["/app/data"]
EXPOSE 3000
CMD ["npm", "start"]
