import os
import pytest


class _DummyPartyBox:
    pass


def test_factory_uses_inmemory_by_default(monkeypatch):
    # Ensure env var is not set
    monkeypatch.delenv('CAMPFIRES_GRAPH_BACKEND', raising=False)

    from campfires.core.factory import CampfireFactory, InMemoryGraphStore

    factory = CampfireFactory(party_box=_DummyPartyBox(), start_background_tasks=False)
    assert isinstance(factory.graph_store, InMemoryGraphStore)


def test_factory_env_neo4j_without_driver_falls_back(monkeypatch):
    # Set backend to neo4j but leave Neo4jGraphStore as None (no driver)
    monkeypatch.setenv('CAMPFIRES_GRAPH_BACKEND', 'neo4j')

    import importlib
    factory_mod = importlib.import_module('campfires.core.factory')
    # Ensure Neo4jGraphStore is None to simulate missing driver
    monkeypatch.setattr(factory_mod, 'Neo4jGraphStore', None, raising=False)

    factory = factory_mod.CampfireFactory(party_box=_DummyPartyBox(), start_background_tasks=False)
    assert isinstance(factory.graph_store, factory_mod.InMemoryGraphStore)


def test_factory_selects_neo4j_when_available(monkeypatch):
    # Set backend to neo4j and patch Neo4jGraphStore to a dummy
    monkeypatch.setenv('CAMPFIRES_GRAPH_BACKEND', 'neo4j')
    monkeypatch.setenv('NEO4J_URI', 'bolt://localhost:7687')
    monkeypatch.setenv('NEO4J_USER', 'neo4j')
    monkeypatch.setenv('NEO4J_PASSWORD', 'pass')
    monkeypatch.setenv('NEO4J_DATABASE', 'neo4j')

    import importlib
    factory_mod = importlib.import_module('campfires.core.factory')

    class _DummyNeo4jStore:
        def __init__(self, uri: str, user: str, password: str, database: str | None = None):
            self.uri = uri
            self.user = user
            self.password = password
            self.database = database

    monkeypatch.setattr(factory_mod, 'Neo4jGraphStore', _DummyNeo4jStore, raising=True)

    factory = factory_mod.CampfireFactory(party_box=_DummyPartyBox(), start_background_tasks=False)
    assert isinstance(factory.graph_store, _DummyNeo4jStore)
    assert factory.graph_store.uri == 'bolt://localhost:7687'
    assert factory.graph_store.user == 'neo4j'
    assert factory.graph_store.password == 'pass'
    assert factory.graph_store.database == 'neo4j'