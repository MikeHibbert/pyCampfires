#!/usr/bin/env python3
"""
Tax Application Development Team Demo

This demo showcases how different developer personas can collaborate using RAG-enabled
knowledge to plan and design a tax application. Each team member brings specialized
expertise to the discussion.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directory to the path so we can import campfires
sys.path.insert(0, str(Path(__file__).parent.parent))

from campfires.core.camper import Camper
from campfires.core.openrouter import LLMCamperMixin, OpenRouterConfig
from campfires.party_box.local_driver import LocalDriver

class TeamMember(LLMCamperMixin, Camper):
    """Base class for team members with specialized RAG knowledge"""
    
    def __init__(self, party_box, config, name, role, rag_file):
        self.name = name
        self.role = role
        # Add RAG file path to config
        config_with_rag = config.copy()
        config_with_rag["rag_document_path"] = rag_file
        super().__init__(party_box, config_with_rag)
        
        # Set up LLM client for actual API calls
        openrouter_config = OpenRouterConfig(
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000)
        )
        self.setup_llm(openrouter_config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Override the prompt to include RAG-enhanced system prompt and generate actual LLM response"""
        try:
            # Get the final system prompt (includes RAG if available)
            final_system_prompt = self.get_system_prompt()
            
            # Create an enhanced prompt that combines the RAG system prompt with the user's question
            enhanced_prompt = f"""You are a {self.role}. {final_system_prompt}

Question: {raw_prompt}

Please provide specific, actionable recommendations for our tax application project based on your expertise. Focus on practical insights that are directly relevant to your role."""
            
            # Make actual LLM call
            response = await self.llm_completion_with_mcp(enhanced_prompt, channel="team_discussion")
            
            return {
                'claim': response,
                'confidence': 0.9,
                'metadata': {
                    'role': self.role,
                    'name': self.name,
                    'has_rag': bool(final_system_prompt),
                    'rag_loaded': bool(final_system_prompt)
                }
            }
        except Exception as e:
            return {
                'claim': f"Error generating response from {self.name}: {str(e)}",
                'confidence': 0.0,
                'metadata': {
                    'error': True,
                    'role': self.role,
                    'name': self.name
                }
            }
    
    async def process_topic(self, topic):
        """Process a topic and return the team member's perspective."""
        # Create a prompt that incorporates the topic with the member's role
        prompt = f"""
        As a {self.role} working on a tax application project, please provide your perspective on:
        
        {topic}
        
        Consider your expertise and focus on aspects most relevant to your role.
        Provide specific, actionable insights.
        """
        
        # Create an input torch with the prompt
        from campfires.core.torch import Torch
        input_torch = Torch(
            claim=prompt,
            source_campfire="tax_app_demo",
            channel="team_discussion"
        )
        
        # Use the process method to get the response
        response_torch = await self.process(input_torch)
        return response_torch.claim

class TaxAppTeamDemo:
    """Orchestrates the team collaboration demo"""
    
    def __init__(self):
        import os
        self.party_box = LocalDriver("demos/party_box")
        self.config = {
            "model": "anthropic/claude-3.5-sonnet",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Get absolute paths for RAG documents
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rag_dir = os.path.join(current_dir, "rag_examples")
        
        # Initialize team members
        self.team_members = {
            "backend": TeamMember(
                self.party_box, 
                self.config, 
                "Alex Chen", 
                "Senior Backend Engineer",
                os.path.join(rag_dir, "backend_developer.yaml")
            ),
            "frontend": TeamMember(
                self.party_box, 
                self.config, 
                "Sarah Johnson", 
                "Senior Frontend Engineer",
                os.path.join(rag_dir, "frontend_developer.json")
            ),
            "testing": TeamMember(
                self.party_box, 
                self.config, 
                "Marcus Rodriguez", 
                "Senior QA Engineer",
                os.path.join(rag_dir, "testing_developer.txt")
            ),
            "devops": TeamMember(
                self.party_box, 
                self.config, 
                "Emily Zhang", 
                "Senior DevOps Engineer",
                os.path.join(rag_dir, "devops_developer.yaml")
            )
        }
        
        self.discussion_log = []
        self.action_items = []
    
    async def conduct_team_meeting(self):
        """Simulate a team planning meeting"""
        print("=" * 80)
        print("🏢 TAX APPLICATION DEVELOPMENT TEAM MEETING")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Objective: Plan and design a comprehensive tax application")
        print()
        
        # Meeting agenda
        agenda_items = [
            {
                "topic": "Project Overview and Requirements",
                "question": "We need to build a comprehensive tax application that helps individuals and small businesses file their taxes. What are the key requirements and considerations from your domain perspective?"
            },
            {
                "topic": "Technical Architecture Discussion", 
                "question": "Based on the requirements, what technical architecture and technology stack would you recommend for this tax application?"
            },
            {
                "topic": "Security and Compliance Considerations",
                "question": "What security measures and compliance requirements should we implement to ensure the tax application meets industry standards?"
            },
            {
                "topic": "User Experience and Interface Design",
                "question": "How should we design the user interface and experience to make tax filing intuitive and accessible for our users?"
            },
            {
                "topic": "Testing Strategy and Quality Assurance",
                "question": "What testing approach should we take to ensure the accuracy and reliability of tax calculations and user workflows?"
            },
            {
                "topic": "Deployment and Operations Strategy",
                "question": "How should we deploy and operate this application to handle tax season traffic and ensure high availability?"
            }
        ]
        
        for i, agenda_item in enumerate(agenda_items, 1):
            print(f"\n📋 AGENDA ITEM {i}: {agenda_item['topic']}")
            print("-" * 60)
            print(f"Question: {agenda_item['question']}")
            print()
            
            # Get input from each team member
            for role, member in self.team_members.items():
                print(f"🗣️  Getting input from {member.name} ({member.role})...")
                try:
                    from campfires.core.torch import Torch
                    input_torch = Torch(
                        claim=agenda_item['question'],
                        source_campfire="tax_app_demo",
                        channel="team_discussion"
                    )
                    response_torch = await member.process(input_torch)
                    response = response_torch.claim
                    print(f"\n{response}\n")
                    
                    self.discussion_log.append({
                        "agenda_item": i,
                        "topic": agenda_item['topic'],
                        "member": member.name,
                        "role": member.role,
                        "response": response
                    })
                    
                except Exception as e:
                    print(f"❌ Error getting response from {member.name}: {e}")
            
            print("-" * 60)
        
        # Generate action plan
        await self.generate_action_plan()
        
        # Generate HTML report
        self.generate_html_report()
    
    async def generate_action_plan(self):
        """Generate a consolidated action plan based on team input"""
        print("\n🎯 GENERATING TEAM ACTION PLAN")
        print("=" * 50)
        
        action_plan_prompt = """Based on the team discussion above, generate a comprehensive action plan for our tax application project:

Please provide a detailed technical action plan including:

1. IMMEDIATE NEXT STEPS (Next 2 weeks):
   - Specific development tasks with owners
   - Infrastructure setup requirements
   - Initial architecture decisions

2. TECHNOLOGY STACK RECOMMENDATIONS:
   - Backend framework and database choices
   - Frontend framework and UI libraries
   - DevOps and deployment tools
   - Testing frameworks and tools
   - Security and compliance tools

3. TECHNICAL ARCHITECTURE DECISIONS:
   - System architecture patterns (microservices vs monolith)
   - Database design approach
   - API design strategy
   - Authentication and authorization approach
   - Data encryption and security measures

4. IMPLEMENTATION TIMELINE:
   - Phase 1: Core infrastructure and basic functionality (Weeks 1-4)
   - Phase 2: Tax calculation engine and forms (Weeks 5-8)
   - Phase 3: Security hardening and compliance (Weeks 9-12)
   - Phase 4: Testing, optimization, and deployment (Weeks 13-16)

5. RISK MITIGATION STRATEGIES:
   - Technical risks and mitigation plans
   - Security vulnerabilities and prevention
   - Performance bottlenecks and solutions
   - Compliance gaps and remediation

6. RESOURCE ALLOCATION:
   - Team member responsibilities
   - External dependencies and integrations
   - Budget considerations for tools and services

Focus on concrete, actionable technical decisions that leverage each team member's expertise."""
        
        # Use the tech consultant for strategic planning, fallback to backend
        plan_generator = None
        for role, member in self.team_members.items():
            if "consultant" in role.lower():
                plan_generator = member
                break
        
        if not plan_generator:
            plan_generator = self.team_members["backend"]
        
        try:
            from campfires.core.torch import Torch
            input_torch = Torch(
                claim=action_plan_prompt,
                source_campfire="tax_app_demo",
                channel="action_planning"
            )
            response_torch = await plan_generator.process(input_torch)
            action_plan = response_torch.claim
            print(action_plan)
            
            self.action_items.append({
                "type": "action_plan",
                "content": action_plan,
                "generated_by": plan_generator.name,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"❌ Error generating action plan: {e}")
    
    def generate_html_report(self):
        """Generate an HTML report of the team meeting and conclusions"""
        print("\n📄 GENERATING HTML REPORT")
        print("=" * 30)
        
        html_content = self._create_html_report()
        
        # Save the report
        report_path = f"demos/tax_app_team_meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ HTML report generated: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            return None
    
    def _create_html_report(self):
        """Create the HTML content for the team meeting report"""
        
        # Generate team member summaries
        team_summary = ""
        for role, member in self.team_members.items():
            team_summary += f"""
            <div class="team-member">
                <h4>{member.name}</h4>
                <p><strong>Role:</strong> {member.role}</p>
                <p><strong>RAG Enabled:</strong> {'✅ Yes' if hasattr(member, '_rag_system_prompt') and member._rag_system_prompt else '❌ No'}</p>
            </div>
            """
        
        # Generate discussion summary
        discussion_summary = ""
        current_topic = ""
        
        for entry in self.discussion_log:
            if entry['topic'] != current_topic:
                if current_topic:
                    discussion_summary += "</div>"
                current_topic = entry['topic']
                discussion_summary += f"""
                <div class="agenda-item">
                    <h3>📋 {entry['topic']}</h3>
                """
            
            response_content = entry['response'].replace('\n', '<br>')
            discussion_summary += f"""
            <div class="member-response">
                <h4>{entry['member']} ({entry['role']})</h4>
                <div class="response-content">{response_content}</div>
            </div>
            """
        
        if current_topic:
            discussion_summary += "</div>"
        
        # Generate action items summary
        action_items_html = ""
        for item in self.action_items:
            action_content = item['content'].replace('\n', '<br>')
            action_items_html += f"""
            <div class="action-item">
                <h4>Action Plan (Generated by {item['generated_by']})</h4>
                <div class="action-content">{action_content}</div>
                <p><small>Generated: {item['timestamp']}</small></p>
            </div>
            """
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tax Application Development Team Meeting Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #667eea;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .team-members {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                }}
                .team-member {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #667eea;
                }}
                .team-member h4 {{
                    margin: 0 0 10px 0;
                    color: #333;
                }}
                .agenda-item {{
                    margin-bottom: 30px;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                .agenda-item h3 {{
                    background: #667eea;
                    color: white;
                    margin: 0;
                    padding: 15px;
                }}
                .member-response {{
                    padding: 20px;
                    border-bottom: 1px solid #f0f0f0;
                }}
                .member-response:last-child {{
                    border-bottom: none;
                }}
                .member-response h4 {{
                    color: #764ba2;
                    margin: 0 0 10px 0;
                }}
                .response-content {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 3px solid #667eea;
                }}
                .action-item {{
                    background: #e8f5e8;
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 4px solid #28a745;
                    margin-bottom: 15px;
                }}
                .action-item h4 {{
                    color: #155724;
                    margin: 0 0 10px 0;
                }}
                .action-content {{
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #666;
                    font-style: italic;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-card h3 {{
                    margin: 0;
                    font-size: 2em;
                }}
                .stat-card p {{
                    margin: 5px 0 0 0;
                    opacity: 0.9;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏢 Tax Application Development Team Meeting</h1>
                <p>📅 {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
                <p>🎯 Objective: Plan and design a comprehensive tax application</p>
            </div>

            <div class="section">
                <h2>📊 Meeting Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <h3>{len(self.team_members)}</h3>
                        <p>Team Members</p>
                    </div>
                    <div class="stat-card">
                        <h3>{len([entry for entry in self.discussion_log])}</h3>
                        <p>Total Responses</p>
                    </div>
                    <div class="stat-card">
                        <h3>{len(set(entry['topic'] for entry in self.discussion_log))}</h3>
                        <p>Agenda Items</p>
                    </div>
                    <div class="stat-card">
                        <h3>{len(self.action_items)}</h3>
                        <p>Action Plans</p>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>👥 Team Members</h2>
                <div class="team-members">
                    {team_summary}
                </div>
            </div>

            <div class="section">
                <h2>💬 Discussion Summary</h2>
                {discussion_summary}
            </div>

            <div class="section">
                <h2>🎯 Action Items & Conclusions</h2>
                {action_items_html}
            </div>

            <div class="footer">
                <p>Generated by Campfires RAG-enabled Team Collaboration Demo</p>
                <p>Powered by specialized developer personas with domain expertise</p>
            </div>
        </body>
        </html>
        """
        
        return html_template

async def main():
    """Run the tax application team demo"""
    print("🚀 Starting Tax Application Development Team Demo")
    print("This demo showcases RAG-enabled developer personas collaborating on a project")
    print()
    
    try:
        demo = TaxAppTeamDemo()
        await demo.conduct_team_meeting()
        
        print("\n✅ Demo completed successfully!")
        print("Check the generated HTML report for a comprehensive summary.")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())