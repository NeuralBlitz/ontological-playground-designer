# API Reference: Ontological Playground Designer

This document provides a comprehensive reference for the Ontological Playground Designer API. This API allows programmatic interaction with the AI's core capabilities, enabling external services, web frontends, and automated workflows to generate and evaluate axiomatically aligned simulated worlds.

---

## 🚀 Base URL

The API is designed to run locally by default during development.

**Local Development Base URL:** `http://localhost:8000`

---

## 🔒 Authentication

Currently, this API does not implement explicit authentication or authorization. For local development and testing, all endpoints are publicly accessible.

**[Future Enhancement]:** Implement API key or OAuth2 authentication for secure deployment.

---

## 💡 Data Models

The following Pydantic models define the structure of data sent to and received from the API.

---

### `AxiomDefinition`

Represents a single axiom (ethical or flourishing principle) that guides the AI's world design.

```json
{
  "id": "string",
  "principle": "string",
  "priority": 99,
  "type": "string",
  "keywords": [
    "string"
  ],
  "enforcement_strategy": "string"
}
```

---

### `AxiomSetRequest`

The request body for generating a new world.

```json
{
  "world_name": "string",
  "axioms": [
    {
      "id": "string",
      "principle": "string",
      "priority": 99,
      "type": "string",
      "keywords": [
        "string"
      ],
      "enforcement_strategy": "string"
    }
  ]
}
```

---

### `GeneratedWorldResponse`

The response body after successfully generating a new world.

```json
{
  "world_name": "string",
  "creation_timestamp": "2023-10-27T10:00:00.000000",
  "output_format": "json",
  "output_file": "string",
  "message": "string"
}
```

---

### `EvaluationMetric`

Represents a single predicted performance metric for a world, linked to specific axioms.

```json
{
  "name": "string",
  "value": 0.0,
  "target_axiom_ids": [
    "string"
  ],
  "interpretation": "string"
}
```

---

### `DetectedParadox`

Details a single identified logical inconsistency or ethical contradiction.

```json
{
  "id": "string",
  "description": "string",
  "type": "string",
  "severity": 0.0,
  "involved_rules_ids": [
    "string"
  ],
  "conflict_path_description": "string",
  "suggested_resolution": "string"
}
```

---

### `WorldEvaluationRequest`

The request body for evaluating an existing world.

```json
{
  "world_name": "string",
  "world_config": {
    "simulation_defaults": {
      "simulation_engine_version": "generic_agent_based_v1.0",
      "initial_world_size": {
        "x_dim": 100,
        "y_dim": 100,
        "z_dim": 1
      },
      "time_step_duration_ms": 100,
      "max_simulation_steps": 50000
    },
    "world_metadata": {
      "name": "string",
      "creation_timestamp": "string",
      "axioms_influencing_design": [
        "string"
      ],
      "designed_by": "string"
    },
    "generated_world_rules": {
      "agent_behaviors": [
        {
          "id": "string",
          "description": "string",
          "parameters": {},
          "dependencies": []
        }
      ],
      "environmental_laws": [],
      "resource_mechanics": [],
      "social_dynamics": [],
      "system_mechanics": [],
      "meta_rules": [],
      "unclassified_rules": []
    }
  },
  "axioms": [
    {
      "id": "string",
      "principle": "string",
      "priority": 99,
      "type": "string",
      "keywords": [
        "string"
      ],
      "enforcement_strategy": "string"
    }
  ]
}
```

---

### `WorldEvaluationResponse`

The response body after successfully evaluating a world.

```json
{
  "world_name": "string",
  "evaluation_timestamp": "2023-10-27T10:00:00.000000",
  "overall_flourishing_score": 0.0,
  "axiom_adherence_scores": {
    "string": 0.0
  },
  "predicted_metrics": [
    {
      "name": "string",
      "value": 0.0,
      "target_axiom_ids": [
        "string"
      ],
      "interpretation": "string"
    }
  ],
  "paradox_risk_score": 0.0,
  "detected_paradoxes": [
    {
      "id": "string",
      "description": "string",
      "type": "string",
      "severity": 0.0,
      "involved_rules_ids": [
        "string"
      ]
    }
  ],
  "recommendations": [
    "string"
  ],
  "message": "string"
}
```

---

## 🌐 Endpoints

---

### `GET /health`

Checks if the API and core components are operational.

*   **Description:** A simple health check endpoint to verify the API server is running and its internal AI components are initialized.
*   **Response:** `HTTP 200 OK` with a status message, or `HTTP 500 Internal Server Error` if components failed to load.

```bash
# Example Request
curl -X GET "http://localhost:8000/health"

# Example Response (Success)
# HTTP/1.1 200 OK
# Content-Type: application/json
{
  "status": "operational",
  "message": "Ontological Playground Designer API is ready."
}
```

---

### `POST /generate_world`

Generates a new axiomatically aligned simulated world based on provided axioms.

*   **Description:** This endpoint triggers the AI's core world-design process. It takes a list of axiomatic principles and returns a confirmation that the world's blueprint has been generated and saved.
*   **Request Body:** [`AxiomSetRequest`](#axiomsetrequest)
*   **Response:** [`GeneratedWorldResponse`](#generatedworldresponse) on success, or `HTTP 500 Internal Server Error` on failure.

```bash
# Example Request
curl -X POST "http://localhost:8000/generate_world" -H "Content-Type: application/json" -d '{
  "world_name": "my_api_designed_world",
  "axioms": [
    {
      "id": "PHILOSOPHY_FLOURISHING_001",
      "principle": "Maximize the long-term well-being of all sentient agents.",
      "priority": 1,
      "type": "ethical"
    },
    {
      "id": "ECOLOGY_SUSTAINABILITY_001",
      "principle": "Ensure all resource consumption rates are perpetually sustainable.",
      "priority": 2,
      "type": "environmental"
    }
  ]
}'

# Example Response (Success)
# HTTP/1.1 201 Created
# Content-Type: application/json
{
  "world_name": "my_api_designed_world",
  "creation_timestamp": "2023-10-27T10:00:00.000000",
  "output_format": "json",
  "output_file": "data/generated_worlds/my_api_designed_world.json",
  "message": "World designed successfully."
}
```

---

### `POST /evaluate_world`

Evaluates an AI-designed world for axiomatic alignment and flourishing potential.

*   **Description:** This endpoint assesses a previously generated world configuration. It predicts its long-term flourishing trajectory, checks for adherence to guiding axioms, and detects any internal paradoxes or inconsistencies.
*   **Request Body:** [`WorldEvaluationRequest`](#worldevaluationrequest)
*   **Response:** [`WorldEvaluationResponse`](#worldevaluationresponse) on success, or `HTTP 500 Internal Server Error` on failure.

```bash
# Example Request
# (Note: `world_config` must be the full JSON content of a file generated by /generate_world)
curl -X POST "http://localhost:8000/evaluate_world" -H "Content-Type: application/json" -d '{
  "world_name": "my_api_designed_world",
  "world_config": {
    "simulation_defaults": {
      "simulation_engine_version": "generic_agent_based_v1.0",
      "initial_world_size": { "x_dim": 100, "y_dim": 100, "z_dim": 1 },
      "max_simulation_steps": 50000
    },
    "world_metadata": {
      "name": "my_api_designed_world",
      "creation_timestamp": "2023-10-27T10:00:00.000000",
      "axioms_influencing_design": ["PHILOSOPHY_FLOURISHING_001", "ECOLOGY_SUSTAINABILITY_001"],
      "designed_by": "Ontological Playground Designer AI"
    },
    "generated_world_rules": {
        "agent_behaviors": [{"id": "Agent_Cooperation_my_api_designed_world_0", "description": "Agents gain well-being from cooperative actions.", "type": "agent_behavior", "parameters": {"cooperation_reward_factor": 0.8}, "dependencies": [], "axiom_influence": {"PHILOSOPHY_FLOURISHING_001": 1.0}}],
        "environmental_laws": [{"id": "Resource_Regen_my_api_designed_world_1", "description": "Resource regeneration rate is tied to current ecological health.", "type": "environmental_law", "parameters": {"regen_health_multiplier": 0.9}, "dependencies": [], "axiom_influence": {"ECOLOGY_SUSTAINABILITY_001": 1.0}}],
        "meta_rules": [{"id": "Rule_Consistency_Check_my_api_designed_world_2", "description": "Meta-rule: All generated rules are checked for internal logical consistency.", "type": "meta_rule", "parameters": {"consistency_tolerance": 0.999}, "dependencies": ["World_Root"], "axiom_influence": {"EPISTEMIC_COHERENCE_001": 1.0}}],
        "resource_mechanics": [], "social_dynamics": [], "system_mechanics": [], "unclassified_rules": []
    }
  },
  "axioms": [
    {
      "id": "PHILOSOPHY_FLOURISHING_001",
      "principle": "Maximize the long-term well-being of all sentient agents.",
      "priority": 1,
      "type": "ethical"
    },
    {
      "id": "ECOLOGY_SUSTAINABILITY_001",
      "principle": "Ensure all resource consumption rates are perpetually sustainable.",
      "priority": 2,
      "type": "environmental"
    },
    {
      "id": "EPISTEMIC_COHERENCE_001",
      "principle": "Maintain absolute logical and conceptual coherence.",
      "priority": 0,
      "type": "foundational"
    }
  ]
}'

# Example Response (Success)
# HTTP/1.1 200 OK
# Content-Type: application/json
{
  "world_name": "my_api_designed_world",
  "evaluation_timestamp": "2023-10-27T10:00:00.000000",
  "overall_flourishing_score": 0.85,
  "axiom_adherence_scores": {
    "PHILOSOPHY_FLOURISHING_001": 0.92,
    "ECOLOGY_SUSTAINABILITY_001": 0.88
  },
  "predicted_metrics": [
    {
      "name": "total_flourishing_score",
      "value": 0.90,
      "target_axiom_ids": ["PHILOSOPHY_FLOURISHING_001"],
      "interpretation": "Predicted overall well-being and adaptive capacity: 0.90 (higher is better)."
    }
  ],
  "paradox_risk_score": 0.05,
  "detected_paradoxes": [],
  "recommendations": [
    "The designed world shows strong axiomatic alignment and flourishing potential. Consider exploring more complex axiom interactions or increasing world scale."
  ],
  "message": "World evaluated successfully."
}
```

---

**[End of `docs/api_reference.md`]**

---

