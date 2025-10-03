# elsciRL Application: A GridWorld Classroom Simulation

<div align="center">
  <b>Application of the elsciRL system to a Classroom Simulator.</b>
  <br>
  Visit our <a href="https://elsci.org">website</a> to get started, explore our <a href="https://github.com/pdfosborne/elsciRL-Wiki">open-source Wiki</a> to learn more or join our <a href="https://discord.gg/GgaqcrYCxt">Discord server</a> to connect with the community.
  <br>
  <i>In pre-alpha Development.</i>
</div>

<div align="center">  
  <br>

  <a href="https://github.com/pdfosborne/elsciRL">![elsciRL GitHub](https://img.shields.io/github/watchers/pdfosborne/elsciRL?style=for-the-badge&logo=github&label=elsciRL&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL)</a>
  <a href="https://github.com/pdfosborne/elsciRL-Wiki">![Wiki GitHub](https://img.shields.io/github/watchers/pdfosborne/elsciRL-Wiki?style=for-the-badge&logo=github&label=elsciRL-Wiki&link=https%3A%2F%2Fgithub.com%2Fpdfosborne%2FelsciRL-Wiki)</a>
  <a href="https://discord.gg/GgaqcrYCxt">![Discord](https://img.shields.io/discord/1310579689315893248?style=for-the-badge&logo=discord&label=Discord&link=https%3A%2F%2Fdiscord.com%2Fchannels%2F1184202186469683200%2F1184202186998173878)</a>

</div>

## Classroom Environment

The objective of this problem is to help the teacher's initiative of recycling.  Specifically, scrap paper must be passed around the classroom and placed into the recycling bin and avoiding the general waste bin.

The agent suggests which direction the student should pass the paper. However, there is a chance that the student may not follow the suggestion thereby making this a probabilistic environment.

### Real-World Safety Concerns

This is a educational and completely artificial problem that was introduced to highlight the challenges of introducing automation with reinforcement learning in real-world problems. 

In this example, the agent's suggestions must not discriminate against students in any way, accidental or otherwise. It must balance acting on observable features without those features being associated with protected characteristics. Therefore, interpretability and human control is required to ensure the automated output of agents are applied safely. 

The reason we introduced this example was because of the issues when training AI models with images or video feed for the classroom as a representation of the states. The dataset used in the current example describes students by their clothing choices, hair style/colour, piercings, etc. No protected characteristics were used nor are any of the students representations of real people. 

The student images were created by providing the student descriptions to an AI image generation to illustrate the real-world aspect of this problem as well as highlighting the potential issues of implementing such automation.

