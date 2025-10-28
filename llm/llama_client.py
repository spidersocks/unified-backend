import boto3
from llm.config import SETTINGS

def call_llama(prompt: str, max_tokens: int = 60, temperature: float = 0.0, stop: list = None) -> str:
    """
    Calls Bedrock Llama 70B instruct with a prompt and returns the output.
    """
    # Use Bedrock runtime API for Llama 70B instruct
    bedrock = boto3.client("bedrock-runtime", region_name=SETTINGS.aws_region)
    model_arn = SETTINGS.kb_model_arn
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        body["stop"] = stop
    resp = bedrock.invoke_model(
        modelId=model_arn,  # e.g., 'meta.llama3-70b-instruct-v1:0'
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = resp["body"].read().decode("utf-8")
    try:
        out = json.loads(result)
        return out.get("generation", "").strip() or out.get("output", "").strip() or out.get("text", "").strip()
    except Exception:
        return result.strip()