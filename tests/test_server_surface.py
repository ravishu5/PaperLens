"""The MCP surface as a whole: primitives used for what they are for."""
import asyncio

from paperlens.server import mcp


def _run(coro):
    # asyncio.run creates and closes its own loop. Creating loops without closing
    # them leaked them into later modules, which run their own event loops.
    return asyncio.run(coro)


def test_every_vision_capability_is_exposed():
    names = {t.name for t in _run(mcp.list_tools())}
    for capability in ("reverse_engineer_paper", "trace_method", "find_implementations",
                       "map_paper_to_code", "compare_paper_with_code",
                       "find_implementation_gaps", "compare_implementations",
                       "trace_research_lineage", "find_sota_successors",
                       "build_reproduction_plan"):
        assert capability in names, capability


def test_no_tool_is_a_thin_api_wrapper():
    """The premise the vision doc rejects: if the surface reads as
    search_arxiv/download_pdf/search_github, the project has failed."""
    names = {t.name for t in _run(mcp.list_tools())}
    for banned in ("search_arxiv", "download_pdf", "search_semantic_scholar",
                   "search_github", "get_citations", "fetch_paper"):
        assert banned not in names


def test_every_tool_documents_itself():
    for t in _run(mcp.list_tools()):
        assert t.description and len(t.description) > 40, t.name


def test_workflows_are_prompts_not_tools():
    """Prompts are user-invoked; tools are model-invoked. The multi-step
    workflows belong in the former."""
    prompts = {p.name for p in _run(mcp.list_prompts())}
    assert prompts == {"reverse-engineer-paper", "reproduce-paper",
                       "compare-implementations", "trace-lineage"}


def test_every_graph_node_kind_is_addressable():
    templates = {t.uri_template for t in _run(mcp.list_resource_templates())}
    for shape in ("paperlens://paper/{arxiv_id}",
                  "paperlens://paper/{arxiv_id}/section/{section_path}",
                  "paperlens://paper/{arxiv_id}/equation/{ref}",
                  "paperlens://paper/{arxiv_id}/component/{slug}",
                  "paperlens://paper/{arxiv_id}/references",
                  "paperlens://repo/{owner}/{repo}/symbol/{+qualified_name}",
                  "paperlens://mapping/{arxiv_id}/{owner}/{repo}",
                  "paperlens://lineage/{arxiv_id}"):
        assert shape in templates, shape


def test_a_fetch_tool_mirrors_the_resource_surface():
    """Resource support across clients is uneven, so the graph must also be
    reachable without it."""
    assert "paperlens_fetch" in {t.name for t in _run(mcp.list_tools())}


def test_uris_that_can_contain_slashes_use_reserved_expansion():
    templates = {t.uri_template for t in _run(mcp.list_resource_templates())}
    assert any("{+path}" in t for t in templates)
    assert any("{+qualified_name}" in t for t in templates)
