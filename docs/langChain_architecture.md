# LangChain architecture

The previous partial-migration description has been replaced by the implemented architecture.

- [Implementation and workflow](architecture.md)
- [Setup and user workflow](../README.md)
- [Deployment, backups and troubleshooting](operations.md)
- [Validation evidence and remaining limits](validation.md)

The AI flow now uses LCEL composition, LangChain message history interfaces, scoped retrieval, structured generation and validated citations. The former astronomy-specific orchestration, shared browser session, mock confidence and CLI-only ingestion have been replaced.
