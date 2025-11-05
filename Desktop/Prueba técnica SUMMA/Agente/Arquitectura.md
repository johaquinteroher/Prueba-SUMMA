```mermaid
graph TD
    % Definición de componentes y flujo de alto nivel
    subgraph Ambito de Ejecución del Agente (Runtime)
        A[1. Agente Orquestador (main.py)] --> B[2. Modelo de Lenguaje (LLM)];
        A --> C[3. Módulo de Herramientas (Tools)];
    end

    subgraph Modulo de Herramientas
        C --> D(Función: get_severance_pay_info);
        C --> E(Función: get_PTO_balance);
    end

    subgraph Recursos Externos
        F[4. Capa de Datos]
    end

    % Flujos de usuario y datos
    User[Usuario Final] --> A;
    D --> F;
    E --> F;

    % Estilos para hacerlo más legible (Opcional)
    style A fill:#A3CCFF,stroke:#337AB7,stroke-width:2px
    style B fill:#CBE6C9,stroke:#4CAF50,stroke-width:2px
    style C fill:#FFD700,stroke:#FFA500,stroke-width:2px
    style D fill:#FFD700,stroke:#FFA500,stroke-width:2px
    style E fill:#FFD700,stroke:#FFA500,stroke-width:2px
    style F fill:#DDA0DD,stroke:#9932CC,stroke-width:2px
    style User fill:#FFB6C1,stroke:#FF69B4,stroke-width:2px
```