.. mermaid::

   graph TD
      classDef client fill:#D5E8D4,stroke:#82B366,color:#000000;
      classDef old_optional fill:#E1D5E7,stroke:#9673A6,color:#000000;
      classDef optional fill:#DAE8FC,stroke:#6C8EBF,color:#000000,stroke-dasharray: 3 3;
      
      ip("Loadbalancer / Public IP")
      smtp("SMTP Mailbox")
      a("Database")
      b("Task Queue")
      c("File Storage")
      worker("Generic Worker (1:n)")
      web("Web & API (1:n)")
      beats("Beats Worker (1:n)")
      mail("Mail-Scan (0:1)")
      
      %% Outside references
      smtp <-- Poll emails --> mail
      ip <--> web
      
      %% Optional Containers
      ocr("OCR (0:n)")
      segmentation("Segmentation (0:n)")
      summarization("Summarization (0:n)")
      flower("Flower (0:1)")
      
      %% Server / Cluster
      h0("Server 1")
      h1("Server 2")
      h2("Server 3")
      h3("Server 4")
      h4("Server 5")
      i("...")
      j("Server with GPU")

      subgraph all["Private Network"]
      subgraph databases["Persistent Container / Services"]
      a
      c
      b
      end
      subgraph containers["Stateless Containers"]
      mail
      web
      flower
      worker
      beats
      subgraph optional["Optional Containers"]
      ocr
      segmentation
      summarization
      end
      end
      subgraph servers["Server / Cluster"]
      h0
      h1
      h2
      h3
      h4
      i
      j
      end    
      worker <--> databases
      worker -- Can delegate tasks--> optional
      worker -- "Can process tasks"--> worker
      web <--> databases
      web <--> flower
      flower <--> b
      mail <--> databases
      beats <--> databases
      containers -- "Operated on"--> servers
      databases -- "Can be operated on"--> servers
      end
      
      click flower "/web/on_premises.html#optional-6-use-flower-to-monitor-tasks"
      click web "/web/on_premises.html#start-the-container"
      click worker "/web/on_premises.html#start-the-container"
      click ocr "/web/on_premises.html#optional-7-use-azure-read-api-on-premise"
      click segmentation "/web/on_premises.html#optional-8-install-document-segmentation-container"
      click summarization "/web/on_premises.html#optional-9-install-document-summarization-container" 
      
      class flower optional
      class ocr optional
      class mail optional
      class ocr optional
      class segmentation optional
      class summarization optional
      class h1 optional
      class h2 optional
      class h3 optional
      class h4 optional
      class i optional
      class j optional

