# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2024-10-20

### Added
- **Version Consistency**: Ensured all version references are properly synchronized
- **Enhanced Package Distribution**: Improved build and distribution process
- **Documentation Updates**: Comprehensive documentation improvements and examples
- **Testing Infrastructure**: Enhanced test coverage and reliability

### Fixed
- **Multimodal API Compatibility**: Addressed compatibility issues in multimodal components
- **Package Metadata**: Improved package configuration and metadata consistency

### Changed
- **Build Process**: Streamlined distribution package creation and validation
- **Release Process**: Enhanced PyPI publishing workflow and documentation

## [0.3.0] - 2024-10-20

### Added
- **YAML Save/Restore Functionality**: Complete campfire configuration persistence
  - Save individual campfire configurations to YAML files
  - Restore campfires from YAML configurations
  - Bulk operations with CampfireManager for multiple campfires
  - Advanced filename templating with timestamps and custom patterns
  - Comprehensive error handling and validation
- **Enhanced Documentation**: 
  - Added YAML Save/Restore section to main README.md with code examples
  - Documented test_yaml_save_restore.py demo in demos/README.md
  - Added comprehensive YAML Configuration Management section to docs/usage_examples.md
  - Included best practices, file structure examples, and advanced templating

### Fixed
- Version consistency: Updated __init__.py to match pyproject.toml version (0.3.0)
- Multimodal demo compatibility: Replaced SVG with PNG for LLaVA compatibility

### Changed
- Improved documentation structure and organization
- Enhanced code examples with practical use cases

## [0.2.0] - 2024-10-19

### Added
- **Enhanced Orchestration System**: Comprehensive execution tracking and reporting
  - Detailed execution stages with timing and metadata
  - Interactive HTML reports with rich visualizations
  - Multi-stage processing with enhanced tracking capabilities
  - RAG integration for context-aware processing
- **LLM Integration Features**:
  - LLMCamperMixin for seamless LLM integration
  - Support for OpenRouter and Ollama providers
  - Custom prompt engineering capabilities
  - RAG-enhanced team collaboration
- **Interactive HTML Reports**:
  - Real-time execution tracking
  - Visual progress indicators
  - Detailed performance metrics
  - Customizable report templates

### Added
- Comprehensive documentation for all new features
- Usage examples and best practices
- Demo scripts showcasing new capabilities

## [0.1.0] - 2024-10-18

### Added
- Initial release of Campfires AI Agent Framework
- Core campfire and camper functionality
- Basic orchestration capabilities
- Multimodal support for text, images, and audio
- Party Box shared storage system
- MCP Protocol for inter-campfire communication
- Zeitgeist opinion analysis engine
- Basic documentation and examples

### Features
- Python 3.8+ support
- Async/await support throughout
- Extensible plugin architecture
- MIT License