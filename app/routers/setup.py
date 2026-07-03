"""Setup router: API key configuration and status."""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm import get_available_providers, create_llm, _key
from app.config import get_voyage_key, get_openai_key, get_password, get_default_provider, PORT

router = APIRouter(prefix="/api", tags=["setup"])


class SetupRequest(BaseModel):
    provider: str
    api_key: str


class ModelSwitchRequest(BaseModel):
    provider: str
    model: str


@router.get("/status")
async def get_status():
    """实时状态——每次都读当前内存配置，不用 import 快照"""
    providers = get_available_providers()
    has_anthropic = bool(_key("anthropic"))
    has_deepseek = bool(_key("deepseek"))
    has_openai = bool(_key("openai"))
    return {
        "configured": has_anthropic or has_deepseek or has_openai,
        "default_provider": get_default_provider(),
        "providers": providers,
        "embedding": "voyage" if get_voyage_key() else ("openai" if get_openai_key() else "bm25"),
        "password_protected": bool(get_password()),
        "port": PORT,
    }


@router.post("/setup")
async def setup_key(req: SetupRequest):
    """保存并校验 API Key——先保存到 .env 和内存，再校验"""
    provider = req.provider.lower().strip()
    key = req.api_key.strip()

    if not key:
        raise HTTPException(400, "API Key 不能为空")

    env_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    if provider not in env_key_map:
        raise HTTPException(400, f"未知提供商: {provider}")

    # 写入 os.environ（llm.py 的 _get_key 每次实时读取这里）
    os.environ[env_key_map[provider]] = key

    # 写入 .env（持久化，下次启动自动加载）
    _write_env(env_key_map[provider], key)

    # 校验
    validate_msg = ""
    try:
        llm = create_llm(provider)
        await llm.chat("Say OK.", [{"role": "user", "content": "Hi"}])
        validate_msg = "校验通过"
    except Exception as e:
        validate_msg = f"已保存但校验失败: {str(e)[:200]}"

    return {"ok": True, "provider": provider, "message": validate_msg}


@router.put("/model")
async def switch_model(req: ModelSwitchRequest):
    return {"ok": True, "provider": req.provider, "model": req.model}


@router.post("/auth")
async def authenticate(password: str = ""):
    if not get_password():
        return {"ok": True}
    if password == get_password():
        return {"ok": True}
    raise HTTPException(401, "密码错误")


def _write_env(key: str, value: str):
    """写入 .env 文件，不存在则创建"""
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").split("\n")
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}")
        env_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
