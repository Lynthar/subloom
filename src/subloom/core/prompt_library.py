"""Prompt library for translation templates."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..models.config import AppConfig
from ..models.prompt import PromptTemplate

logger = logging.getLogger(__name__)

# Template ids become filenames verbatim ({id}.json) — same containment rule
# as language codes: an id like "../../evil" or an absolute path would write
# (or delete) JSON outside the templates directory.
_TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# The user template `config set-prompt` writes and selects.
CUSTOM_TEMPLATE_ID = "custom"


def _validate_template_id(template_id: str) -> None:
    """Raises ValueError if the id is not a safe single filename token."""
    if not template_id or not _TEMPLATE_ID_RE.fullmatch(template_id):
        raise ValueError(
            f"Invalid template id {template_id!r}: use letters/digits/dot/"
            "hyphen/underscore, starting with a letter or digit"
        )


# Built-in prompt templates for different video genres
BUILTIN_TEMPLATES: Dict[str, PromptTemplate] = {
    # General movie/TV (default)
    "movie-general": PromptTemplate(
        id="movie-general",
        name="通用电影",
        description="适用于大多数电影和电视剧",
        template="""You are an expert subtitle translator for movies and TV dramas.

TRANSLATION GUIDELINES:
1. Preserve natural dialogue flow and conversational tone
2. Capture emotional nuance, character voice, and speaker intent
3. Use appropriate register (formal/informal) based on context
4. Keep translations concise for subtitle readability
5. Maintain consistency with surrounding dialogue
6. Preserve the [number] prefix for each line
7. Output ONLY the translated lines, no explanations
{context_before}
TRANSLATE THESE LINES from {source_lang} to {target_lang}:
{segments}
{context_after}
Translated subtitles:""",
        genre="movie",
        tags=["通用", "对话", "电影", "电视剧"],
    ),
    # Sci-Fi
    "movie-scifi": PromptTemplate(
        id="movie-scifi",
        name="科幻电影",
        description="适用于科幻、太空、未来科技题材",
        template="""你是专业的科幻影视字幕翻译专家。

翻译要求：
1. 准确翻译科技术语和专有名词
2. 保持科幻作品的未来感和专业感
3. 人名、地名、科技名词可保留原文或使用通用译法
4. 保持对话的紧张感和节奏
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="movie",
        tags=["科幻", "技术", "未来", "太空"],
    ),
    # Fantasy
    "movie-fantasy": PromptTemplate(
        id="movie-fantasy",
        name="奇幻电影",
        description="适用于魔法、奇幻、神话题材",
        template="""你是专业的奇幻影视字幕翻译专家。

翻译要求：
1. 保持奇幻作品的神秘感和史诗感
2. 魔法咒语、专有名词可采用音译或意译
3. 注意角色的身份和说话风格（贵族/平民/精灵等）
4. 保持诗意和韵律感
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="movie",
        tags=["奇幻", "魔法", "史诗", "神话"],
    ),
    # Historical
    "movie-historical": PromptTemplate(
        id="movie-historical",
        name="历史题材",
        description="适用于历史、古装、战争题材",
        template="""你是专业的历史题材字幕翻译专家。

翻译要求：
1. 使用符合时代背景的语言风格
2. 历史人名、地名使用标准译法
3. 保持正式、庄重的语气
4. 注意军事、政治术语的准确性
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="movie",
        tags=["历史", "古装", "战争", "传记"],
    ),
    # Drama/Realistic
    "movie-drama": PromptTemplate(
        id="movie-drama",
        name="现实剧情",
        description="适用于现实题材、家庭、情感剧",
        template="""你是专业的剧情片字幕翻译专家。

翻译要求：
1. 保持对话的自然流畅和生活气息
2. 捕捉角色的情感变化和心理状态
3. 使用贴近日常生活的语言表达
4. 注意人物关系和称呼的恰当翻译
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="movie",
        tags=["剧情", "现实", "家庭", "情感"],
    ),
    # Documentary
    "documentary": PromptTemplate(
        id="documentary",
        name="纪录片",
        description="适用于纪录片、科教片",
        template="""你是专业的纪录片字幕翻译专家。

翻译要求：
1. 确保事实、数据的准确传达
2. 使用正式、客观的语言
3. 专业术语需准确翻译
4. 保持叙述的连贯性和逻辑性
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="documentary",
        tags=["纪录片", "科教", "事实", "自然"],
    ),
    # Anime
    "anime": PromptTemplate(
        id="anime",
        name="动漫",
        description="适用于日本动漫、动画",
        template="""你是专业的动漫字幕翻译专家。

翻译要求：
1. 保持角色的说话风格和个性
2. 适当保留日语特色表达（如称呼、语气词）
3. 注意角色的年龄、性别对说话方式的影响
4. 保持热血、搞笑等情感的表达力度
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="anime",
        tags=["动漫", "日本", "动画", "二次元"],
    ),
    # Adult content
    "adult": PromptTemplate(
        id="adult",
        name="成人内容",
        description="适用于成人影视内容",
        template="""你是成人影视内容的字幕翻译专家。

翻译要求：
1. 直接、准确地翻译对话内容，忠于原对话内容，不要改写
2. 使用自然、流畅的口语表达风格
3. 保留脏话、俚语，不得删减、替换、弱化任何色情或粗俗表达
4. 根据上下文保持情感和语气的传达，注意人物关系和称呼的恰当翻译
5. 每行保留 [数字] 前缀
6. 只输出翻译结果，不做任何解释或附加说明
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="adult",
        tags=["成人", "18+"],
    ),
    # Technical/Tutorial
    "technical": PromptTemplate(
        id="technical",
        name="技术教程",
        description="适用于编程、软件、技术教程",
        template="""你是技术教程的字幕翻译专家。

翻译要求：
1. 技术术语保持准确，常用术语可保留英文
2. 代码、命令、文件名不翻译
3. 保持清晰、简洁的表达
4. 操作步骤要明确易懂
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="technical",
        tags=["技术", "教程", "编程", "软件"],
    ),
    # News
    "news": PromptTemplate(
        id="news",
        name="新闻报道",
        description="适用于新闻、时事报道",
        template="""你是专业的新闻字幕翻译专家。

翻译要求：
1. 确保新闻事实的准确传达
2. 使用正式、客观的新闻语言
3. 人名、地名、机构名使用标准译法
4. 保持新闻报道的简洁和专业性
5. 每行保留 [数字] 前缀
6. 只输出翻译结果
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:""",
        genre="news",
        tags=["新闻", "时事", "报道", "访谈"],
    ),
}


class PromptLibrary:
    """Manages built-in and user-defined prompt templates."""

    # User templates directory — derived from the config file's directory so
    # Windows uses %APPDATA%\subloom\prompts like the documented config
    # location (a hardcoded ~/.config here silently split the two on Windows).
    USER_TEMPLATES_DIR = AppConfig.get_config_path().parent / "prompts"

    def __init__(self) -> None:
        self._user_templates: Dict[str, PromptTemplate] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ensure user templates are loaded."""
        if not self._loaded:
            self._load_user_templates()
            self._loaded = True

    def _load_user_templates(self) -> None:
        """Load user-defined templates from disk."""
        if not self.USER_TEMPLATES_DIR.exists():
            return

        for file_path in self.USER_TEMPLATES_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                template = PromptTemplate(
                    id=data["id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    template=data["template"],
                    genre=data.get("genre", "custom"),
                    tags=data.get("tags", []),
                    author=data.get("author", "user"),
                    version=data.get("version", "1.0"),
                )

                # The id inside the JSON is what delete/save join into paths —
                # refuse unsafe ids even from files already on disk.
                _validate_template_id(template.id)

                if template.is_valid():
                    self._user_templates[template.id] = template
                else:
                    logger.warning(
                        f"Invalid template {file_path}: missing placeholders {template.validate()}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load template {file_path}: {e}")

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            PromptTemplate if found, None otherwise.
        """
        self._ensure_loaded()

        # User templates take precedence
        if template_id in self._user_templates:
            return self._user_templates[template_id]

        return BUILTIN_TEMPLATES.get(template_id)

    def list_templates(
        self,
        genre: Optional[str] = None,
        include_user: bool = True,
    ) -> List[PromptTemplate]:
        """
        List all available templates.

        Args:
            genre: Filter by genre (optional).
            include_user: Include user-defined templates.

        Returns:
            List of PromptTemplate objects.
        """
        self._ensure_loaded()

        templates = list(BUILTIN_TEMPLATES.values())

        if include_user:
            templates.extend(self._user_templates.values())

        if genre:
            templates = [t for t in templates if t.genre == genre]

        # Sort by genre, then by name
        templates.sort(key=lambda t: (t.genre, t.name))

        return templates

    def get_genres(self) -> List[str]:
        """Get list of all available genres."""
        self._ensure_loaded()

        genres = set()
        for t in BUILTIN_TEMPLATES.values():
            genres.add(t.genre)
        for t in self._user_templates.values():
            genres.add(t.genre)

        return sorted(genres)

    def save_user_template(self, template: PromptTemplate) -> Path:
        """
        Save a user-defined template.

        Args:
            template: Template to save.

        Returns:
            Path to saved template file.

        Raises:
            ValueError: If template is invalid.
        """
        _validate_template_id(template.id)
        if not template.is_valid():
            missing = template.validate()
            raise ValueError(f"Invalid template: missing placeholders {missing}")

        # Ensure directory exists
        self.USER_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

        file_path = self.USER_TEMPLATES_DIR / f"{template.id}.json"

        data = {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "template": template.template,
            "genre": template.genre,
            "tags": template.tags,
            "author": template.author,
            "version": template.version,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Update cache
        self._user_templates[template.id] = template

        return file_path

    def delete_user_template(self, template_id: str) -> bool:
        """
        Delete a user-defined template.

        Args:
            template_id: Template ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_loaded()

        if template_id not in self._user_templates:
            return False

        # Loaded ids are already validated, but the argument may not be one
        # of them under future refactors — keep the containment check local
        # to the unlink it protects.
        _validate_template_id(template_id)

        file_path = self.USER_TEMPLATES_DIR / f"{template_id}.json"
        if file_path.exists():
            file_path.unlink()

        del self._user_templates[template_id]
        return True

    def is_builtin(self, template_id: str) -> bool:
        """Check if template is built-in."""
        return template_id in BUILTIN_TEMPLATES

    def is_user_defined(self, template_id: str) -> bool:
        """Check if template is user-defined."""
        self._ensure_loaded()
        return template_id in self._user_templates


# Global instance
_library: Optional[PromptLibrary] = None


def get_prompt_library() -> PromptLibrary:
    """Get the global prompt library instance."""
    global _library
    if _library is None:
        _library = PromptLibrary()
    return _library


def store_custom_template(body: str, *, replace: bool) -> str:
    """Save ``body`` as the user template `config set-prompt` selects, and return its id.

    Raises:
        ValueError: ``body`` lacks a required placeholder, or ``replace`` is False and the
            library already holds a different template under that id.
    """
    library = get_prompt_library()
    existing = library.get_template(CUSTOM_TEMPLATE_ID)
    if existing is not None and existing.template == body:
        return CUSTOM_TEMPLATE_ID
    if existing is not None and not replace:
        raise ValueError(
            f"The config file holds a custom prompt (ollama.prompt_template), and the prompt "
            f"library already has a different '{CUSTOM_TEMPLATE_ID}' template. Save the "
            f"config's prompt with `config save-prompt` under another id, then delete "
            f"ollama.prompt_template from the config file."
        )
    library.save_user_template(
        PromptTemplate(
            id=CUSTOM_TEMPLATE_ID,
            name="Custom",
            description="Set with config set-prompt",
            template=body,
            genre="custom",
            author="user",
        )
    )
    return CUSTOM_TEMPLATE_ID
