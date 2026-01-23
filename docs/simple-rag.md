```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'background': '#000000',
      'primaryColor': '#000000',
      'primaryTextColor': '#ffffff',
      'primaryBorderColor': '#ffffff',
      'lineColor': '#ffffff',
      'clusterBkg': 'none',
      'clusterBorder': '#ffffff',
      'titleColor': '#ffffff',
      'edgeLabelBackground': '#000000'
    },
    'flowchart': { 'nodeSpacing': 30, 'rankSpacing': 60, 'curve': 'basis' }
  }
}%%
graph TD
    classDef default fill:none,stroke:#fff,stroke-width:2px,color:#fff;
    subgraph Main_Execution ["<b>Main.py Workflow</b>"]
        direction TB
        Start((Start))
        step_0["rag_config = YamlFile.load()"]
        step_1["prompts_template = YamlFile.load()"]
        step_2["docs = OpenRagBenchJSON.load()"]
        step_3["embedding = LangChainEmbeddingModel()"]
        step_4["chunker = LangChainRecursive()"]
        step_5["vectorstore = LangChainChroma()"]
        End((End))
    end
    Start --> step_0
    step_0 --> step_1
    step_1 --> step_2
    step_2 --> step_3
    step_3 --> step_4
    step_4 --> step_5
    step_5 --> End
    step_0 -.-> YamlFile
    step_1 -.-> YamlFile
    step_2 -.-> OpenRagBenchJSON
    step_3 -.-> LangChainEmbeddingModel
    step_4 -.-> LangChainRecursive
    step_5 -.-> LangChainChroma
    LangChainEmbeddingModel --> BaseEmbeddingModel
    LangChainChroma --> BaseVectorDB
    LangChainRecursive --> BaseChunker
    OpenRagBenchJSON --> BaseDataLoader
```