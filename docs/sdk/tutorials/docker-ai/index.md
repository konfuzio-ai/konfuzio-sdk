## Using Docker AIs with Konfuzio Server

This is an upcoming feature and not yet available.

This tutorial will walk you through the process of using Dockerized Artificial Intelligence models (AI) with Konfuzio Server. 
We will be looking at a sequence diagram involving four components:

Generic Worker
Kubernetes Agent
Custom AI (Dockerized AI)
Container Registry

The sequence is as follows:

.. mermaid::

    sequenceDiagram
        participant GenericWorker as Generic Worker
        participant KubernetesAgent as Kubernetes Agent
        participant CustomAI as Custom AI
        participant ContainerRegistry as Container Registry
    
        GenericWorker->>KubernetesAgent: Request Custom AI
        KubernetesAgent->>ContainerRegistry: Pull Image
        KubernetesAgent->>CustomAI: Start Container
        GenericWorker->>CustomAI: API Call
        CustomAI->>CustomAI: Run AI
        CustomAI->>GenericWorker: API Response
        GenericWorker->>KubernetesAgent: Terminate Custom AI
        KubernetesAgent->>CustomAI: Stop Container
        

.. mermaid::

    graph TD
        a("Fast API (/extract)")
        b("Konfuzio SDK")
        c("Custom AI")
        subgraph all["Konfuzio Server"]
        f("Generic Worker")
        d("Kubernetes Agent")
        f -- "API Call" --> a
        f --> d
        d <-- "Start/Stop" --> containers
        d <-- "Pull Image" --> registry
        subgraph registry["Container Registry"]
        end  
        subgraph containers["Customer Container"]
        a <--> b
        b <--> c
        end     
        end

--

To provide a better understanding, here's an improved explanation of the components and their interactions:

Generic Worker: This component represents the client or application that initiates the interaction with the Custom AI. It sends a request to the Kubernetes Agent for a specific Custom AI service.

Kubernetes Agent: The Kubernetes Agent handles the orchestration of Custom AI services. It receives the request from the Generic Worker and interacts with the Container Registry to fetch the Docker image containing the required AI model. Once the image is retrieved, the Kubernetes Agent starts the Custom AI container.

Custom AI: This component encapsulates the Dockerized AI model. It receives API calls from the Generic Worker, performs the AI processing, and sends back the API response.

Container Registry: The Container Registry stores the Docker images of various Custom AI models. When requested by the Kubernetes Agent, it pulls the specified Docker image and provides it for starting the Custom AI container.

The integration between these components allows the Generic Worker to make API calls to the Custom AI model running within a Docker container, leveraging the capabilities of Konfuzio Server.

Feel free to explore this tutorial and adapt it to your specific use case.