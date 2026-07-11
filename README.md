# 🩺 Diabetes Prediction MLOps Pipeline

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

**Building ML models is cool. But deploying them in production? That's where the real magic happens! ✨**

[Live Demo](#) • [Documentation](#) • [Report Bug](#)

</div>

---

## 🎯 What's This All About?

Hey there! 👋 Welcome to my **end-to-end MLOps project** where I took a diabetes prediction model from my laptop all the way to production on Google Cloud. 

This isn't just another "train a model" project. This is the **real deal** - complete with Docker containers, automated CI/CD pipelines, and Kubernetes orchestration. Everything you'd see in a production ML system at a tech company!

### 💡 Why This Project Stands Out

- 🚀 **Production-Ready**: Not a toy project - this is how real companies deploy ML
- 🔄 **Full Automation**: Push code → Automatic build → Deploy to cloud (no manual steps!)
- 🏗️ **Scalable Architecture**: Kubernetes means it can handle 10 users or 10,000
- 🛡️ **Secure**: No hardcoded credentials, everything uses proper cloud authentication
- 📚 **Well-Documented**: Because good code deserves good docs

---

## 🏗️ The Big Picture

Here's how everything fits together:

```mermaid
graph LR
    A[👨‍💻 You Push Code] --> B[🔄 GitHub Actions]
    B --> C[🐳 Build Docker Image]
    C --> D[📦 GCP Artifact Registry]
    D --> E[☸️ Deploy to GKE]
    E --> F[🌐 Live FastAPI Service]
    F --> G[😊 Happy Users]
```

**In simple English:** You push code → GitHub automatically builds it → Uploads to Google Cloud → Deploys to Kubernetes → Users can access your ML API!

---

## 🧠 The Machine Learning Bit

**Problem I'm Solving:** Predicting diabetes risk based on patient health data

**What Goes In:**
- Age, BMI, blood pressure, glucose levels, etc.

**What Comes Out:**
- Prediction: "High risk" or "Low risk" for diabetes

**The Model:** Trained using scikit-learn (the actual `.pkl` file isn't in the repo for size reasons - it gets loaded from cloud storage)

> 💭 **Fun fact**: The model part was actually the easy bit. The deployment infrastructure? That's where I learned the most!

---

## 📂 How It's Organized

```
📦 diabetes-mlops-pipeline
┣ 📂 app/
┃ ┣ 📜 main.py           # The FastAPI magic happens here
┃ ┣ 📜 schemas.py        # Request/response validation
┃ ┗ 📜 utils.py          # Prediction logic & model loading
┃
┣ 📂 k8s/
┃ ┣ 📜 deployment.yaml   # Kubernetes deployment config
┃ ┗ 📜 service.yaml      # Kubernetes service config
┃
┣ 📂 .github/workflows/
┃ ┗ 📜 ci-cd.yml         # The automation pipeline
┃
┣ 🐳 Dockerfile          # Container recipe
┣ 📋 requirements.txt    # Python dependencies
┗ 📖 README.md           # You are here!
```

---

## 🚀 Getting Started

### 🏃‍♂️ Run Locally (Quick Start)

Want to see it in action on your machine? Here's how:

```bash
# Clone the repo
git clone https://github.com/yourusername/diabetes-mlops-pipeline.git
cd diabetes-mlops-pipeline

# Install dependencies
pip install -r requirements.txt

# Fire it up!
uvicorn app.main:app --reload
```

Now open your browser and go to: **http://localhost:8000/docs**

You'll see a beautiful interactive API documentation (thanks FastAPI! 🙏)

### 🐳 Run with Docker

Prefer containers? I got you:

```bash
# Build the image
docker build -t diabetes-api .

# Run it
docker run -p 8000:8000 diabetes-api
```

Same deal - visit **http://localhost:8000/docs** and you're good to go!

---

## 🔄 The CI/CD Magic Explained

Okay, here's where things get **really cool**. This is the automation that makes me look like a DevOps wizard! 🧙‍♂️

Every time I push code to the `main` branch, GitHub Actions takes over and does ALL the heavy lifting. No clicking around in cloud consoles, no manual deployments, no SSH-ing into servers. Just pure automation bliss!

### 📁 The Brain of the Operation

Everything happens in this file:
```
.github/workflows/deploy-fastapi-gke.yml
```

This little YAML file is like having a **personal deployment assistant** that never sleeps, never makes mistakes, and works faster than any human ever could.

### 🎬 What Actually Happens?

Here's the **blow-by-blow breakdown** of what goes down when I push code:

```mermaid
graph TD
    A[🎯 You: git push] --> B[⚡ GitHub: Workflow Triggered!]
    B --> C[🔨 Step 1: Build Docker Image]
    C --> D[📦 Step 2: Push to Artifact Registry]
    D --> E[🔐 Step 3: Authenticate to GKE]
    E --> F[🚀 Step 4: Deploy to Kubernetes]
    F --> G[✅ Step 5: Verify Rollout]
    G --> H[🎉 Done! App is Live!]
```

**My favorite part?** I literally just type `git push` and go grab coffee ☕. By the time I'm back, the new version is live in production!

---

## 🎭 Behind the Scenes: The Two-Act Play

The workflow is split into **two main jobs** that run one after another:

### 🎬 Act 1: Build & Ship

**What's happening:** Building the Docker container and sending it to Google Cloud

**The Process:**
1. 📥 **Checkout** - Grabs the latest code
2. 🔐 **Login to GCP** - "Hey Google, it's me!" (using service account)
3. 🐳 **Docker Setup** - Connects Docker to Artifact Registry
4. 🔨 **Build Image** - Packages everything into a container
5. 🚢 **Push to Cloud** - Uploads it to Google's container storage

**Cool Detail:** Each image gets tagged with the Git commit SHA. So if something breaks, I know **exactly** which code version caused it!

```
us-central1-docker.pkg.dev/my-project/mlops-test/diabetes-api:a1b2c3d
                                                                  ↑
                                          This is the exact commit that was deployed!
```

### 🎬 Act 2: Deploy & Verify

**What's happening:** Taking that shiny new container and deploying it to Kubernetes

**The Process:**
1. 🔐 **Authenticate Again** - Double-checking credentials (security first!)
2. 🔌 **Install GKE Plugin** - Modern auth for Kubernetes
3. 🎫 **Get Cluster Access** - "Let me into the cluster, please!"
4. 📋 **Apply Manifests** - Tell Kubernetes about the new version
5. ⏱️ **Wait for Rollout** - Making sure everything actually works

**The Safety Net:** The pipeline actually **waits and watches** to make sure the deployment succeeds. If something goes wrong, the workflow fails and I get notified. No silent failures here!

```bash
kubectl rollout status deployment/diabetes-api
# It literally watches: "Waiting for deployment to complete... 1 of 3 updated replicas..."
# Only marks as ✅ when everything is confirmed working
```

---

## 🔐 The Secret Sauce (GitHub Secrets)

For this automation magic to work, I had to tell GitHub how to access Google Cloud. But I'm not crazy enough to put credentials directly in my code! 😅

Here's what I added as **GitHub Secrets** (Settings → Secrets → Actions):

| Secret Name      | What It Does                                      | Example Value           |
|------------------|---------------------------------------------------|-------------------------|
| `GCP_PROJECT_ID` | Tells GitHub which Google Cloud project to use   | `my-mlops-project-2024` |
| `GKE_CLUSTER`    | Which Kubernetes cluster to deploy to             | `diabetes-api-cluster`  |
| `GCP_SA_KEY`     | The VIP pass to access everything (JSON encoded) | `eyJhbGc...` (base64)   |

> 🔒 **Security Note:** These secrets are encrypted by GitHub and never appear in logs. Even I can't see them after setting them up!

---

## 🛡️ Why This Setup is Secure AF

Let me geek out for a second about security, because I'm pretty proud of this:

- ✅ **Zero Hardcoded Credentials** - Everything uses secrets
- ✅ **Principle of Least Privilege** - Service account has minimal permissions
- ✅ **Immutable Image Tags** - Can't accidentally overwrite images
- ✅ **Encrypted Secrets** - GitHub encrypts everything at rest
- ✅ **Audit Trail** - Every deployment is logged and traceable

**Translation:** Even if someone got access to my repo, they couldn't access my cloud resources. And if they somehow did, I'd know exactly what happened and when!

---

## 🎯 The Deployment Strategy (Zero Downtime Baby!)

Here's something cool: when a new version deploys, **users don't notice a thing**. No downtime, no "We're upgrading, come back in 10 minutes" messages.

**How?** Kubernetes does a **Rolling Update**:

1. 🟢 Old version is running (3 pods)
2. 🆕 Spin up 1 new pod
3. ✅ New pod is healthy? Great!
4. 🔄 Switch traffic to new pod
5. 🔴 Kill 1 old pod
6. ⏪ Repeat until all pods are new

If something goes wrong at step 3? Kubernetes just keeps the old version running. **Automatic rollback!**

```bash
# Watch it happen in real-time
kubectl rollout status deployment/diabetes-api

# Waiting for deployment "diabetes-api" rollout to finish: 1 out of 3 new replicas updated...
# Waiting for deployment "diabetes-api" rollout to finish: 2 out of 3 new replicas updated...
# Waiting for deployment "diabetes-api" rollout to finish: 3 out of 3 new replicas updated...
# deployment "diabetes-api" successfully rolled out
```

---

## 🐛 When Things Go Wrong (They Sometimes Do)

Real talk: not every deployment is perfect. Here's how I debug:

```bash
# Check if pods are running
kubectl get pods
# NAME                           READY   STATUS    RESTARTS   AGE
# diabetes-api-xxxxxxxxx-xxxxx   1/1     Running   0          2m

# Pod in CrashLoopBackOff? Check the logs!
kubectl logs diabetes-api-xxxxxxxxx-xxxxx

# Want even more detail?
kubectl describe pod diabetes-api-xxxxxxxxx-xxxxx

# Check the deployment itself
kubectl describe deployment diabetes-api

# Find the public URL
kubectl get service diabetes-api-service
```

**Pro tip:** 90% of deployment issues are either:
- Environment variables not set correctly
- Image tag mismatch
- Resource limits too low

---

## 💡 What I Learned Building This

Building this CI/CD pipeline taught me **way more** than I expected:

### 📚 Technical Skills
- **GitHub Actions syntax** - YAML can be your friend (once you get past the indentation errors!)
- **Docker multi-stage builds** - Smaller images = faster deployments
- **Kubernetes networking** - Services, ingresses, and how traffic actually flows
- **GCP IAM** - Service accounts, roles, and the principle of least privilege

### 🧠 Soft Skills
- **Patience** - Debugging YAML indentation at 2 AM builds character
- **Documentation** - Future me is grateful present me wrote good docs
- **Security thinking** - Always asking "what could go wrong?"

### 🤔 Biggest Aha Moments

**"Wait, this actually works?!"** - The first time I pushed code and saw it automatically deploy to production was honestly magical

**"Kubernetes is complicated... but worth it"** - The learning curve is steep, but the payoff in scalability and reliability is huge

**"DevOps is just as important as ML"** - A model that can't be deployed easily is just a Jupyter notebook gathering dust

---

## 🔮 What's Coming Next?

This setup is already production-grade, but I'm always thinking about improvements:

### 🎯 Short-term Goals
- [ ] **Slack Notifications** - Get pinged when deployments succeed/fail
- [ ] **Automated Testing** - Run tests before deploying
- [ ] **Image Scanning** - Check for vulnerabilities with Trivy

### 🚀 Long-term Dreams
- [ ] **Multi-environment Setup** - Dev, staging, prod with promotion workflow
- [ ] **Canary Deployments** - Roll out to 10% of users first
- [ ] **Workload Identity** - More secure than service account keys
- [ ] **ArgoCD** - GitOps-style deployments
- [ ] **Auto-cleanup** - Delete old Docker images automatically

---

## 📊 By The Numbers

Just some fun stats about this project:

- 📦 **~50 MB** - Size of the Docker image
- ⚡ **~3 minutes** - Average deployment time
- 🔄 **~30** - Successful deployments so far
- 🐛 **~10** - Failed deployments (we learn from failures!)
- ☕ **∞** - Cups of coffee consumed while debugging

---

## 🤝 Want to Contribute?

Found a bug? Have an idea? Feel free to:
- 🐛 Open an issue
- 🔧 Submit a pull request
- 💡 Share your feedback
- ⭐ Star the repo if you found this helpful!

All contributions are welcome!

---

## 👨‍💻 About Me

**Rohit Dusane**  
Data Scientist | MLOps Enthusiast | Healthcare AI

I'm passionate about building ML systems that actually make it to production. This project combines my interests in machine learning, cloud infrastructure, and healthcare technology.

**Currently exploring:** Feature stores, experiment tracking, and how to make ML systems even more reliable in production.

<div align="center">

[![Gmail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:addb.asst@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/rohit-dusane)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RohitDusane)

</div>

---

## 💬 Final Thoughts

Building this project was a journey from "I can train models" to "I can deploy and maintain production ML systems." The automation piece especially felt like gaining superpowers - pushing code and watching it automatically test, build, and deploy is genuinely exciting even after doing it dozens of times!

If you're learning MLOps, my advice: **just start building**. You'll make mistakes, things will break (a lot), and you'll spend hours debugging YAML indentation. But that's exactly how you learn!

---

<div align="center">

### ⭐ If you found this helpful, drop a star! It keeps me motivated to build cool stuff.

### 🤔 Questions? Feel free to reach out!

**Built with ❤️, lots of ☕, and a healthy dose of 🤦‍♂️ (debugging moments)**



# BEFORE ANY CODE RUN MLFLOW AT LOCAL USING

```bash
mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns `
  --host 0.0.0.0 `
  --port 5000



mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```



### FOR DOCKER AND DEPLOYMENT - LOCALALY TESTING WE NEED

mlflow server `
   --backend-store-uri sqlite:///mlflow.db `
   --default-artifact-root ./mlruns `
   --host 0.0.0.0 `
   --port 5000 `
   --allowed-hosts "host.docker.internal,localhost,127.0.0.1"


</div>