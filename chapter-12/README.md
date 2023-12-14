# Kubernetes

Below is a high-level diagram outlining the steps to
deploy a simple web app using Kubernetes.

<img
  src="https://cdn.mlx.institute/assets/cortex-webapp.png"
  style="border: 1px solid grey; width: 100%;"
/>

<img
  src="https://cdn.mlx.institute/assets/k3s-steps.png"
  style="border: 1px solid grey; width: 100%;"
/>

## Building the WebApp

First what you need to do is add all the files of the
webapp into a folder, and turn it into a docker image

To do that, you'll create a directory named `web`, and
create the files:

* `index.html`
* `main.css`
* `main.js`
* `server.js`

```html
<!-- ./web/index.html -->

<head>
  <link rel="stylesheet" href="main.css">
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Digits recognizer</h1>
    </div>
    <div class="number-drawing">
      <canvas id="number-drawing" width="280" height="280" style="border:1px solid">
        Your browser does not support the HTML5 canvas tag.
      </canvas>
      <textarea id="results"></textarea>
    </div>
    <input value="http://127.0.0.1:8080/predictions/mnist" class="endpoint" id="endpoint" />
    <div class="actions">
      <button id="clear" type="reset">Clear</button>
      <button id="detect" type="submit">Detect</button>
    </div>
  </div>
  <script src="main.js"></script>
</body>
```


```css
/* ./web/main.css */

body {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
  color: #000000;
}

.container {
  display: flex;
  flex-direction: column;
  height: 100%;
  margin-left: 10px;
}

.header {
  height: 60px;
}

.number-drawing {
  display: flex;
  flex-direction: row;
}

#number-drawing {
  background-color: #000000;
}

#results {
  width: 282px;
  height: 282px;
  margin-left: 20px;
}

.endpoint {
  width: 300px;
  margin: 20px 0px;
}

```


```js
// ./web/main.js

const canvas = document.getElementById('number-drawing');
const ctx = canvas.getContext('2d');
ctx.strokeStyle = '#fff';
ctx.fillStyle = '#000';


let isPainting = false;
let lineWidth = 15;
let startX;
let startY;
let offsetY = 60;
let offsetX = 10;


addEventListener('click', async e => {

  if (e.target.id === 'clear') {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  if (e.target.id === 'detect') {
    const cnv = document.createElement('canvas');
    cnv.width = 28; cnv.height = 28;
    const cnv_ctx = cnv.getContext('2d');
    cnv_ctx.drawImage(canvas, 0, 0, cnv.width, cnv.height);
    const img = cnv.toDataURL('image/png', 0.5);
    const endpoint = document.getElementById('endpoint').value;
    const data = { endpoint, img };
    const opts = { method: 'POST', body: JSON.stringify(data) };
    const res = await fetch('/predict', opts);
    const json = await res.json();
    document.getElementById('results').innerHTML = JSON.stringify(json, null, 2);
  }
});


const draw = (e) => {
  if (!isPainting) return;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineTo(e.clientX - offsetX, e.clientY - offsetY);
  ctx.stroke();
}


canvas.addEventListener('mousedown', (e) => {
  isPainting = true;
  startX = e.clientX;
  startY = e.clientY;
});


canvas.addEventListener('mouseup', e => {
  isPainting = false;
  ctx.stroke();
  ctx.beginPath();
});


canvas.addEventListener('mousemove', draw);
```


```js
// ./web/server.js

const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const PORT = 3456;


function parse(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('error', (error) => reject(error));
    req.on('data', chunk => body += chunk.toString());
    req.on('end', () => resolve(JSON.parse(body)));
  });
}


const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(__dirname, 'main.css'), 'utf8');
const js = fs.readFileSync(path.join(__dirname, 'main.js'), 'utf8');


const handlers = async (req, res) => {

  const { pathname } = url.parse(req.url, true);
  const method = req.method;
  console.log(`${method} ${pathname}`);

  if (pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    return res.end(html);
  }

  if (pathname === '/main.css') {
    res.writeHead(200, { 'Content-Type': 'text/css' });
    return res.end(css);
  }

  if (pathname === '/main.js') {
    res.writeHead(200, { 'Content-Type': 'text/javascript' });
    return res.end(js);
  }

  if (pathname === '/predict' && method === 'POST') {
    const body = await parse(req);
    console.log(`body`, body);
    const headers = { 'Content-Type': 'application/json' };
    const opts = { method: 'POST', headers, body: JSON.stringify(body) };
    const response = await fetch(body.endpoint, opts);
    const json = await response.json();
    console.log(`json`, json);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(json));
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
};


const server = http.createServer(handlers);
server.listen(PORT, () => { console.log(`Server running on ${PORT}`) });
```

With the files above, you have the web app created. You could
open `index.html` using your browser and it should display it.

But we need to create an docker image out of it, and upload
it to [Docker Hub](https://hub.docker.com/) So you'll also
need to create an account and a repo there. Name the repo `minst-web-app`

```sh
# ./web/Dockerfile

# Use an official Node runtime as a parent image
FROM node:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 3456

# Define environment variable
ENV NODE_ENV production

# Run server.js when the container launches
CMD ["node", "server.js"]
```
Now login to dockerhub using the docker command
```sh
docker login
```

And build with 
```sh
docker buildx build --platform linux/amd64 -t [account-name]/minst-web-app:1.0 . --push
```
**Replace account name with your username**

Lastly, we need to deploy it on Kubernetes. We declare the
resources that we want using the yaml below:

```yaml
# node-pod-web-app.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mnist-web-app
  labels:
    name: mnist-web-app
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mnist-web-app
  namespace: mnist-web-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mnist-web-app
  template:
    metadata:
      labels:
        app: mnist-web-app
    spec:
      containers:
      - name: mnist-web-app
        image: [account-name]/minst-web-app:1.0
        imagePullPolicy: Always
        ports:
        - containerPort: 3456
---
apiVersion: v1
kind: Service
metadata:
  name: mnist-web-app-service
  namespace: mnist-web-app
spec:
  selector:
    app: mnist-web-app
  type: NodePort
  ports:
    - protocol: TCP
      port: 80
      targetPort: 3456
```
And lastly you apply and create the resources

```sh
kubectl apply -f node-pod-web-app.yaml
```