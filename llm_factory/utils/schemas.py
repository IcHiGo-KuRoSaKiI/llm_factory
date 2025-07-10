"""Common JSON schemas used across llm_factory utilities."""

components_relationships_schema = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node identifier"},
                    "name": {"type": "string", "description": "Human readable name"},
                    "description": {"type": "string", "description": "Optional details about the component"}
                },
                "required": ["id", "name"],
                "additionalProperties": False
            }
        },
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "ID of the source node"},
                    "target": {"type": "string", "description": "ID of the target node"},
                    "description": {"type": "string", "description": "Relationship details"}
                },
                "required": ["source", "target"],
                "additionalProperties": False
            }
        }
    },
    "required": ["nodes", "connections"],
    "additionalProperties": False
}
