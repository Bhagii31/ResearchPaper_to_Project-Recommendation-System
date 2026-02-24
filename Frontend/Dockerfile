FROM node:20-alpine
WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# ✅ Add these lines
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

RUN npm run build
CMD ["npm", "run", "start", "--", "-p", "3000"]