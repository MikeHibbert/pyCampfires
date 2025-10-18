"""
Hospital Administration Zeitgeist Demo

This demo showcases five campers with distinct roles in patient administration 
at a small hospital, each using Zeitgeist to research topics from their unique perspectives.

Characters:
- Sarah (Head Nurse): Warm but frazzled, empathetic, hates paperwork
- Tom (Admin Coordinator): Analytical, data-driven, systems-focused
- Priya (Patient Advocate): Fiery, patient-focused, calls out inefficiencies
- Liam (IT Specialist): Quiet, tech-focused, dreams of digital solutions
- Dr. Elena (Ward Manager): Pragmatic, strategic, balances budgets and care
"""

import asyncio
import logging
import sys
import os
import re
import html
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the path so we can import campfires
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from campfires import Camper, LLMCamperMixin
from campfires.zeitgeist import ZeitgeistEngine, ZeitgeistConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HospitalCamper(LLMCamperMixin, Camper):
    """Base class for hospital staff campers with Zeitgeist capabilities"""
    
    def __init__(self, name: str, role: str, personality: str, concerns: List[str], **kwargs):
        # Initialize with minimal required config for Camper
        party_box = kwargs.get('party_box', None)
        config = kwargs.get('config', {'name': name})
        super().__init__(party_box, config)
        self.role = role
        self.personality = personality
        self.concerns = concerns
        self.zeitgeist_engine = ZeitgeistEngine()
    
    async def override_prompt(self, raw_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Override the base prompt method required by Camper abstract class.
        
        Args:
            raw_prompt: The raw prompt to process
            
        Returns:
            Dictionary containing the processed response
        """
        # Check if this is a personal outlook request
        if "reflect on your current outlook and self-perception" in raw_prompt:
            return await self._generate_personal_outlook_response(raw_prompt)
        
        # For other prompts, return the prompt structure
        # Use system_prompt from RAG document if provided, otherwise use default
        default_system = f"You are {self.role} with the following personality: {self.personality}"
        final_system = system_prompt if system_prompt else default_system
        
        return {
            'system': final_system,
            'user': raw_prompt,
            'metadata': {
                'role': self.role,
                'concerns': self.concerns,
                'personality': self.personality,
                'rag_system_prompt_used': bool(system_prompt)
            }
        }
    
    async def _generate_personal_outlook_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a realistic personal outlook response based on role and personality."""
        
        # Extract topic from prompt if present
        topic = ""
        if "perspective on:" in prompt:
            topic = prompt.split("perspective on:")[-1].strip()
        
        # Generate role-specific outlook responses
        outlook_templates = {
            'head_nurse': {
                'base': "As a head nurse, I lead patient care coordination with a focus on clinical excellence. My {personality} approach ensures comprehensive care delivery.",
                'knowledge': "I have extensive clinical experience and understand the complexities of patient flow and staff management.",
                'concerns': "My primary focus areas are {concerns_text}. I prioritize evidence-based care and effective team coordination.",
                'confidence': "I'm confident in my clinical leadership abilities and my team's capacity to deliver quality care.",
                'topic_view': "Regarding {topic}, I approach it from a patient-first perspective, considering how it impacts care quality and clinical workflows."
            },
            'admin_coordinator': {
                'base': "As an administrative coordinator, I focus on the operational efficiency of our healthcare system. My {personality} approach helps me optimize processes.",
                'knowledge': "I have deep knowledge of hospital workflows, data systems, and process optimization.",
                'concerns': "I'm primarily concerned with {concerns_text}. Data accuracy and system efficiency are my key priorities.",
                'confidence': "I'm very confident in my analytical abilities and process improvement skills.",
                'topic_view': "For {topic}, I immediately think about metrics, workflows, and how we can measure and improve outcomes."
            },
            'patient_advocate': {
                'base': "As a patient advocate, I champion patient rights and healthcare accessibility. My {personality} approach ensures patient-centered care.",
                'knowledge': "I understand patient rights, healthcare accessibility, and the challenges patients face navigating our system.",
                'concerns': "My focus is on {concerns_text}. I work to identify and address barriers to quality patient care.",
                'confidence': "I'm confident in my advocacy skills and my ability to identify patient-centered solutions.",
                'topic_view': "When it comes to {topic}, I evaluate how this serves our patients and advances equitable healthcare delivery."
            },
            'it_specialist': {
                'base': "As an IT specialist, I maintain our digital infrastructure and support healthcare technology initiatives. My {personality} approach focuses on reliable solutions.",
                'knowledge': "I have expertise in healthcare IT systems, integration challenges, and digital workflow optimization.",
                'concerns': "I focus on {concerns_text}. System reliability and effective user adoption are my primary objectives.",
                'confidence': "I'm confident in my technical abilities and my understanding of healthcare technology needs.",
                'topic_view': "Looking at {topic}, I consider how technology can enhance efficiency and support clinical workflows."
            },
            'ward_manager': {
                'base': "As a ward manager, I balance clinical excellence with operational effectiveness. My {personality} approach guides strategic decision-making.",
                'knowledge': "I understand both the clinical and business sides of healthcare, with experience in resource management.",
                'concerns': "I focus on {concerns_text}. Every decision must support both patient care and operational sustainability.",
                'confidence': "I'm confident in my leadership abilities and strategic planning skills.",
                'topic_view': "Regarding {topic}, I consider the implementation requirements and alignment with our strategic objectives."
            }
        }
        
        # Get template for this role
        template = outlook_templates.get(self.role, outlook_templates['head_nurse'])
        
        # Format concerns text
        concerns_text = ", ".join(self.concerns)
        
        # Build response
        response_parts = [
            template['base'].format(personality=self.personality),
            template['knowledge'],
            template['concerns'].format(concerns_text=concerns_text),
            template['confidence']
        ]
        
        if topic:
            response_parts.append(template['topic_view'].format(topic=topic))
        
        response_text = " ".join(response_parts)
        
        return {
            'claim': response_text,
            'confidence': 0.9,
            'metadata': {
                'role': self.role,
                'concerns': self.concerns,
                'personality': self.personality,
                'response_type': 'personal_outlook'
            }
        }
        
    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a topic from this camper's perspective"""
        try:
            # Get general zeitgeist information
            zeitgeist_info = await self.zeitgeist_engine.get_zeitgeist(topic)
            
            # Get role-specific opinions
            role_opinions = await self.zeitgeist_engine.get_role_opinions(topic, self.role)
            
            # Get trending tools/solutions
            trending_tools = await self.zeitgeist_engine.get_trending_tools(topic)
            
            return {
                'zeitgeist': zeitgeist_info,
                'role_opinions': role_opinions,
                'trending_tools': trending_tools,
                'camper_perspective': self._add_personal_perspective(topic, zeitgeist_info)
            }
        except Exception as e:
            logger.error(f"Error researching topic '{topic}' for {self.name}: {e}")
            return {'error': str(e)}
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        """Add personal perspective based on camper's personality and role"""
        # This would typically use the LLM to generate a response
        # For demo purposes, we'll return a template response
        return f"As a {self.role}, I'm particularly interested in how {topic} affects {', '.join(self.concerns)}."
    
    async def share_insight(self, topic: str, research_data: Dict) -> str:
        """Share insights about a topic in character"""
        perspective = research_data.get('camper_perspective', '')
        return f"[{self.name}]: {perspective}"


class SarahHeadNurse(HospitalCamper):
    """Sarah - Head Nurse: Experienced clinical leader focused on patient care coordination"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Sarah",
            role="head_nurse",
            personality="experienced clinical leader, patient-focused, detail-oriented",
            concerns=["patient care quality", "staff coordination", "clinical protocols", "care continuity"],
            **kwargs
        )
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        return f"From a nursing leadership perspective, {topic} needs to be evaluated for its impact on patient care quality and staff workflow. I'm particularly interested in how this integrates with our current clinical protocols and whether it enhances care coordination. The implementation should support our nursing staff while maintaining our high standards of patient care. I'd want to see evidence of improved patient outcomes and staff efficiency before full adoption."


class TomAdminCoordinator(HospitalCamper):
    """Tom - Admin Coordinator: Analytical, data-driven, systems-focused"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Tom",
            role="admin_coordinator",
            personality="analytical, loves data, systems-focused",
            concerns=["efficiency metrics", "cost analysis", "process optimization", "data accuracy"],
            **kwargs
        )
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        return f"Looking at the data on {topic}, I need to see the ROI metrics and implementation timeline. What's the cost-benefit analysis? How does this integrate with our existing EHR system? I've run the numbers on similar initiatives - we need at least 15% efficiency improvement to justify the resource allocation. The system should just work seamlessly."


class PriyaPatientAdvocate(HospitalCamper):
    """Priya - Patient Advocate: Dedicated to patient rights and healthcare accessibility"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Priya",
            role="patient_advocate",
            personality="dedicated advocate, patient-centered, quality-focused",
            concerns=["patient rights", "healthcare accessibility", "care equity", "patient experience"],
            **kwargs
        )
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        return f"From a patient advocacy perspective, {topic} must be evaluated for its direct impact on patient experience and outcomes. I'm particularly concerned with how this affects healthcare accessibility and whether it addresses existing disparities in care. Any implementation should prioritize patient rights and ensure that vulnerable populations aren't disadvantaged. I need to see concrete evidence that this improves patient satisfaction and reduces barriers to quality care."


class LiamITSpecialist(HospitalCamper):
    """Liam - IT Specialist: Quiet, tech-focused, dreams of digital solutions"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Liam",
            role="it_specialist",
            personality="quiet, tech-focused, solution-oriented",
            concerns=["system integration", "digital workflows", "automation", "user adoption"],
            **kwargs
        )
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        return f"From an IT perspective, {topic} presents several technical implementation opportunities. I would need to evaluate integration with our existing systems, particularly our EHR platform. Key considerations include user interface design, system security, HIPAA compliance, and staff adoption strategies. The technical architecture should prioritize reliability, scalability, and ease of use. I'd recommend a phased implementation approach with comprehensive user training and ongoing technical support."


class DrElenaWardManager(HospitalCamper):
    """Dr. Elena - Ward Manager: Strategic medical leader focused on operational excellence"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Dr. Elena",
            role="ward_manager",
            personality="strategic leader, evidence-based, operationally focused",
            concerns=["resource optimization", "clinical outcomes", "operational efficiency", "strategic planning"],
            **kwargs
        )
    
    def _add_personal_perspective(self, topic: str, research_data: Dict) -> str:
        return f"From a strategic management perspective, {topic} requires careful evaluation of both clinical and operational impacts. I need to assess how this aligns with our departmental goals and resource allocation. The implementation must demonstrate measurable improvements in patient outcomes while maintaining cost-effectiveness. I'm particularly interested in the evidence base supporting this initiative and how it integrates with our long-term strategic objectives. Any new program must show clear ROI and sustainable implementation pathways."


async def generate_outlook_comparison_report(staff, pre_outlooks: Dict, post_outlooks: Dict, topic: str):
    """Generate a report comparing pre and post-zeitgeist outlooks"""
    
    print(f"Analyzing how zeitgeist research influenced perspectives on '{topic}':\n")
    
    for camper in staff:
        name = camper.name
        print(f"[ANALYSIS] {name} ({camper.role}):")
        
        # Get pre and post outlook data
        pre_data = pre_outlooks.get(name, {})
        post_data = post_outlooks.get(name, {})
        
        if 'error' in pre_data or 'error' in post_data:
            print(f"   [WARNING] Unable to analyze outlook changes due to errors")
            continue
        
        # Extract responses for comparison
        pre_response = pre_data.get('outlook_response', {})
        post_response = post_data.get('outlook_response', {})
        
        # Extract actual text content for analysis
        if isinstance(pre_response, dict):
            # Check for generated outlook response with 'claim'
            if 'claim' in pre_response:
                pre_text = pre_response['claim']
            else:
                pre_text = pre_response.get('user', str(pre_response))
        else:
            pre_text = str(pre_response) if pre_response else ""
            
        if isinstance(post_response, dict):
            # Check for generated outlook response with 'claim'
            if 'claim' in post_response:
                post_text = post_response['claim']
            else:
                post_text = post_response.get('user', str(post_response))
        else:
            post_text = str(post_response) if post_response else ""
        
        # Basic comparison metrics
        pre_length = len(pre_text.split())
        post_length = len(post_text.split())
        
        print(f"   [METRICS] Response complexity: {pre_length} -> {post_length} words")
        
        # Check for key indicators of change
        confidence_indicators = ['confident', 'certain', 'sure', 'know', 'understand']
        concern_indicators = ['concern', 'worry', 'challenge', 'problem', 'issue']
        solution_indicators = ['solution', 'approach', 'strategy', 'method', 'way']
        
        pre_confidence = sum(1 for word in confidence_indicators if word in pre_text.lower())
        post_confidence = sum(1 for word in confidence_indicators if word in post_text.lower())
        
        pre_concerns = sum(1 for word in concern_indicators if word in pre_text.lower())
        post_concerns = sum(1 for word in concern_indicators if word in post_text.lower())
        
        pre_solutions = sum(1 for word in solution_indicators if word in pre_text.lower())
        post_solutions = sum(1 for word in solution_indicators if word in post_text.lower())
        
        # Report changes
        if post_confidence > pre_confidence:
            print(f"   [+] Increased confidence indicators ({pre_confidence} -> {post_confidence})")
        elif post_confidence < pre_confidence:
            print(f"   [-] Decreased confidence indicators ({pre_confidence} -> {post_confidence})")
        else:
            print(f"   [=] Stable confidence level ({pre_confidence})")
            
        if post_solutions > pre_solutions:
            print(f"   [SOLUTIONS] More solution-oriented language ({pre_solutions} -> {post_solutions})")
        elif post_solutions < pre_solutions:
            print(f"   [SOLUTIONS] Less solution-focused ({pre_solutions} -> {post_solutions})")
            
        if post_concerns > pre_concerns:
            print(f"   [CONCERNS] Increased concern indicators ({pre_concerns} -> {post_concerns})")
        elif post_concerns < pre_concerns:
            print(f"   [CONCERNS] Reduced concern indicators ({pre_concerns} -> {post_concerns})")
        
        # Zeitgeist influence assessment
        if abs(post_length - pre_length) > 10:
            print(f"   [ZEITGEIST] Significant zeitgeist influence detected (major response change)")
        elif abs(post_length - pre_length) > 5:
            print(f"   [ZEITGEIST] Moderate zeitgeist influence detected")
        else:
            print(f"   [ZEITGEIST] Minimal zeitgeist influence detected")
        
        print()


def sanitize_filename(topic: str) -> str:
    """Convert a topic string into a safe filename"""
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', topic)
    # Replace spaces with underscores
    safe_name = re.sub(r'\s+', '_', safe_name)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    # Limit length
    if len(safe_name) > 50:
        safe_name = safe_name[:50].rstrip('_')
    return safe_name


def extract_human_readable_content(data) -> str:
    """Extract human-readable content from complex data structures"""
    def clean_response_text(text: str) -> str:
        """Clean response text by removing prompt instructions"""
        # Remove common prompt patterns
        lines = text.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like prompt instructions
            if any(pattern in line.lower() for pattern in [
                'please provide your honest self-assessment',
                'how you see yourself in your role',
                'your current knowledge and expertise',
                'your main concerns and priorities',
                'your confidence in handling',
                'your perspective on the topic',
                'be authentic and specific',
                'covering:'
            ]):
                skip_next = True
                continue
            
            # Skip numbered list items that are part of prompts
            if line.startswith(('1.', '2.', '3.', '4.', '5.')) and any(word in line.lower() for word in ['how', 'your', 'role', 'knowledge', 'concerns', 'confidence', 'perspective']):
                continue
                
            if line and not skip_next:
                cleaned_lines.append(line)
            elif line:
                skip_next = False
                
        return ' '.join(cleaned_lines)
    
    if isinstance(data, str):
        return clean_response_text(data)
    elif isinstance(data, dict):
        # Handle specific outlook response structure
        if 'outlook_response' in data:
            outlook_response = data['outlook_response']
            if isinstance(outlook_response, dict):
                # Try to extract the actual content from the outlook response
                if 'claim' in outlook_response:
                    return clean_response_text(str(outlook_response['claim']))
                elif 'user' in outlook_response:
                    return clean_response_text(str(outlook_response['user']))
                elif 'content' in outlook_response:
                    return clean_response_text(str(outlook_response['content']))
                elif 'response' in outlook_response:
                    return clean_response_text(str(outlook_response['response']))
                else:
                    # Extract any meaningful text from the outlook response
                    text_parts = []
                    for key, value in outlook_response.items():
                        if isinstance(value, str) and len(value.strip()) > 20:
                            text_parts.append(clean_response_text(value))
                    return ' '.join(text_parts) if text_parts else str(outlook_response)
            else:
                return clean_response_text(str(outlook_response))
        
        # Try to find other human-readable content
        elif 'content' in data:
            return clean_response_text(str(data['content']))
        elif 'response' in data:
            return clean_response_text(str(data['response']))
        elif 'claim' in data:
            return clean_response_text(str(data['claim']))
        elif 'user' in data:
            return clean_response_text(str(data['user']))
        elif 'message' in data:
            return clean_response_text(str(data['message']))
        else:
            # If it's a dict with multiple keys, try to extract meaningful text
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str) and len(value.strip()) > 20 and key not in ['timestamp', 'camper_name', 'role', 'topic']:
                    text_parts.append(clean_response_text(value))
            return ' '.join(text_parts) if text_parts else str(data)
    elif isinstance(data, list):
        # Join list items if they're strings
        text_parts = [clean_response_text(str(item)) for item in data if str(item).strip()]
        return ' '.join(text_parts)
    else:
        return clean_response_text(str(data))

def format_text_for_html(text: str) -> str:
    """Format text for HTML display with proper paragraph breaks"""
    if not text:
        return ""
    
    # Clean up the text
    text = text.strip()
    
    # Split into paragraphs and clean each one
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Join paragraphs with HTML breaks
    if len(paragraphs) > 1:
        return '</p><p>'.join(paragraphs)
    else:
        return text

def generate_html_report(topic: str, staff: List, pre_outlooks: Dict, post_outlooks: Dict, 
                        research_insights: List, discussion_summary: str, 
                        action_proposals: Dict = None, action_items: List = None) -> str:
    """Generate an HTML report of the hospital meeting discussion"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_topic = html.escape(topic)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Meeting Report: {safe_topic}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .staff-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .staff-card {{
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 20px;
            background: #fafbfc;
        }}
        .staff-name {{
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 5px;
        }}
        .staff-role {{
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
        }}
        .outlook-section {{
            margin: 15px 0;
        }}
        .outlook-label {{
            font-weight: bold;
            color: #444;
            margin-bottom: 5px;
        }}
        .outlook-content {{
            background: white;
            padding: 15px;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .pre-zeitgeist {{
            border-left-color: #ff6b6b;
        }}
        .post-zeitgeist {{
            border-left-color: #4ecdc4;
        }}
        .insights {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid #28a745;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e1e8ed;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            margin-top: 0;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Hospital Staff Meeting Report</h1>
            <div class="subtitle">Discussion Topic: {safe_topic}</div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📋 Meeting Overview</h2>
                <p><strong>Topic:</strong> {safe_topic}</p>
                <p><strong>Participants:</strong> {len(staff)} hospital staff members</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>👥 Staff Perspectives</h2>
                <div class="staff-grid">
"""
    
    # Add staff perspectives
    for camper in staff:
        name = html.escape(camper.name)
        role = html.escape(camper.role)
        
        pre_outlook = pre_outlooks.get(camper.name, {})
        post_outlook = post_outlooks.get(camper.name, {})
        
        # Extract human-readable content from outlooks
        pre_content = extract_human_readable_content(pre_outlook)
        post_content = extract_human_readable_content(post_outlook)
        
        # Format content for HTML display
        pre_formatted = format_text_for_html(pre_content)
        post_formatted = format_text_for_html(post_content)
        
        # Escape HTML but preserve our paragraph formatting
        pre_safe = html.escape(pre_formatted)
        post_safe = html.escape(post_formatted)
        
        html_content += f"""
                    <div class="staff-card">
                        <div class="staff-name">{name}</div>
                        <div class="staff-role">{role}</div>
                        
                        <div class="outlook-section">
                            <div class="outlook-label">Initial Perspective:</div>
                            <div class="outlook-content pre-zeitgeist">
                                <p>{pre_safe}</p>
                            </div>
                        </div>
                        
                        <div class="outlook-section">
                            <div class="outlook-label">After Research:</div>
                            <div class="outlook-content post-zeitgeist">
                                <p>{post_safe}</p>
                            </div>
                        </div>
                    </div>
"""
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Research Insights</h2>
                <div class="insights">
"""
    
    # Add research insights
    for i, insight in enumerate(research_insights, 1):
        # Extract and format insight content
        insight_content = extract_human_readable_content(insight)
        insight_formatted = format_text_for_html(insight_content)
        safe_insight = html.escape(insight_formatted)
        
        html_content += f"""
                    <div style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 5px; border-left: 4px solid #28a745;">
                        <h4 style="margin-top: 0; color: #28a745;">Research Insight {i}</h4>
                        <p>{safe_insight}</p>
                    </div>
"""
    
    html_content += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Discussion Summary</h2>
                <div class="summary-box">
                    <h3>Key Outcomes</h3>
                    <p>{html.escape(discussion_summary)}</p>
                </div>
            </div>"""
    
    # Add action items section if available
    if action_items and len(action_items) > 0:
        html_content += f"""
            
            <div class="section">
                <h2>🎯 Action Items</h2>
                <div class="action-items">
                    <h3>Proposed Actions</h3>
"""
        
        for i, item in enumerate(action_items, 1):
            safe_action = html.escape(item['action'])
            safe_proposer = html.escape(item['proposed_by'])
            safe_role = html.escape(item['role'])
            
            html_content += f"""
                    <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;">
                        <div style="font-weight: bold; color: #007bff; margin-bottom: 5px;">Action Item {i}</div>
                        <p style="margin: 5px 0;">{safe_action}</p>
                        <small style="color: #6c757d;">Proposed by: {safe_proposer} ({safe_role})</small>
                    </div>
"""
        
        html_content += """
                </div>
            </div>"""
    
    # Add action proposals section if available
    if action_proposals:
        html_content += f"""
            
            <div class="section">
                <h2>💡 Action Proposals by Role</h2>
                <div class="action-proposals">
"""
        
        for name, proposal_data in action_proposals.items():
            if 'proposal' in proposal_data:
                safe_name = html.escape(name)
                safe_role = html.escape(proposal_data['role'])
                proposal_content = extract_human_readable_content(proposal_data['proposal'])
                proposal_formatted = format_text_for_html(proposal_content)
                safe_proposal = html.escape(proposal_formatted)
                
                html_content += f"""
                    <div style="margin-bottom: 20px; padding: 15px; background: white; border-radius: 5px; border-left: 4px solid #ffc107;">
                        <h4 style="margin-top: 0; color: #ffc107;">{safe_name} ({safe_role})</h4>
                        <p>{safe_proposal}</p>
                    </div>
"""
        
        html_content += """
                </div>
            </div>"""
    
    html_content += f"""
            
            <div class="timestamp">
                Report generated on {timestamp} by Hospital Zeitgeist Demo
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content


async def run_action_planning_stage(topic: str, staff: list, discussion_summary: str, research_insights: list):
    """Run an action planning stage where staff discuss what to do based on the discussion summary"""
    
    print(f"\n{'='*60}")
    print(f"ACTION PLANNING STAGE: '{topic}'")
    print(f"{'='*60}")
    
    print(f"\n📋 DISCUSSION SUMMARY REVIEW")
    print(f"{'='*50}")
    print("The following summary and insights were generated from our discussion:")
    print(f"\nSummary: {discussion_summary}")
    
    print(f"\n🔍 Key Research Insights:")
    for i, insight in enumerate(research_insights, 1):
        insight_content = extract_human_readable_content(insight)
        print(f"{i}. {insight_content[:200]}..." if len(insight_content) > 200 else f"{i}. {insight_content}")
    
    print(f"\n💡 ACTION PLANNING DISCUSSION")
    print(f"{'='*50}")
    print("Now let's discuss: What specific actions can we take to address these findings?")
    
    # Create action planning context
    action_context = f"""
    Based on our discussion about '{topic}', we have identified key insights and reached some agreements.
    
    Discussion Summary: {discussion_summary}
    
    Key Research Insights:
    {chr(10).join([f"- {extract_human_readable_content(insight)}" for insight in research_insights])}
    
    Now we need to move from discussion to action. What specific, actionable steps can we take to address these findings?
    """
    
    # Each staff member proposes actions based on the summary
    action_proposals = {}
    for camper in staff:
        print(f"\n[{camper.name}] Proposing actions for '{topic}':")
        try:
            # Use the action context as a research topic to get action-oriented insights
            action_research = await camper.research_topic(f"actionable solutions for {topic} in hospital setting")
            action_proposal = await camper.share_insight(f"action plan for {topic}", action_research)
            action_proposals[camper.name] = {
                'proposal': action_proposal,
                'research': action_research,
                'role': camper.role
            }
            
            # Extract and display the action proposal
            proposal_content = extract_human_readable_content(action_proposal)
            print(f"  {proposal_content}")
            
        except Exception as e:
            print(f"  Error getting action proposal: {e}")
            action_proposals[camper.name] = {'error': str(e), 'role': camper.role}
    
    print(f"\n🎯 CONSOLIDATING ACTION ITEMS")
    print(f"{'='*50}")
    
    # Generate consolidated action items (this could be enhanced with AI summarization)
    action_items = []
    for name, proposal_data in action_proposals.items():
        if 'proposal' in proposal_data:
            proposal_content = extract_human_readable_content(proposal_data['proposal'])
            # Extract actionable items (simple heuristic - look for action words)
            action_words = ['implement', 'create', 'establish', 'develop', 'train', 'review', 'update', 'improve']
            sentences = proposal_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in action_words) and len(sentence) > 20:
                    action_items.append({
                        'action': sentence,
                        'proposed_by': name,
                        'role': proposal_data['role']
                    })
    
    print("Consolidated Action Items:")
    for i, item in enumerate(action_items, 1):
        print(f"{i}. {item['action']} (Proposed by: {item['proposed_by']} - {item['role']})")
    
    return action_proposals, action_items


async def run_hospital_discussion(topic: str):
    """Run a discussion among hospital staff about a specific topic"""
    
    print(f"\n{'='*60}")
    print(f"HOSPITAL STAFF MEETING: Discussing '{topic}'")
    print(f"{'='*60}")
    
    # Create our hospital staff (party_box and config are optional for this demo)
    staff = [
        SarahHeadNurse(party_box=None, config={}),
        TomAdminCoordinator(party_box=None, config={}),
        PriyaPatientAdvocate(party_box=None, config={}),
        LiamITSpecialist(party_box=None, config={}),
        DrElenaWardManager(party_box=None, config={})
    ]
    
    print(f"\nStaff present:")
    for camper in staff:
        print(f"- {camper.name} ({camper.role}): {camper.personality}")
    
    print(f"\n📋 Meeting Topic: {topic}")
    
    # Capture pre-zeitgeist outlooks
    print(f"\n🧠 INITIAL OUTLOOKS (Before Zeitgeist Research)")
    print(f"{'='*50}")
    pre_outlooks = {}
    for camper in staff:
        print(f"\n[{camper.name}] Initial perspective on '{topic}':")
        try:
            outlook = await camper.get_personal_outlook(topic)
            pre_outlooks[camper.name] = outlook
            # Extract the main response from the outlook
            response = outlook.get('outlook_response', {})
            if isinstance(response, dict):
                # Check if this is a generated outlook response with 'claim'
                if 'claim' in response:
                    claim = response['claim']
                    print(f"  {claim}")
                else:
                    # Fallback to user prompt for non-outlook responses
                    user_prompt = response.get('user', '')
                    print(f"  Prompt: {user_prompt[:200]}..." if len(user_prompt) > 200 else f"  Prompt: {user_prompt}")
            else:
                print(f"  Response: {response}")
        except Exception as e:
            print(f"  Error getting outlook: {e}")
            pre_outlooks[camper.name] = {'error': str(e)}
    
    print(f"\n🔍 ZEITGEIST RESEARCH PHASE")
    print(f"{'='*50}")
    print(f"Each staff member now researches the topic using zeitgeist...\n")
    
    # Each staff member researches the topic
    research_results = {}
    research_insights = []
    for camper in staff:
        print(f"🔎 {camper.name} is researching...")
        research_data = await camper.research_topic(topic)
        research_results[camper.name] = research_data
        
        # Share their perspective
        insight = await camper.share_insight(topic, research_data)
        research_insights.append(f"{camper.name}: {insight}")
        print(f"\n{insight}\n")
    
    # Capture post-zeitgeist outlooks
    print(f"\n🧠 UPDATED OUTLOOKS (After Zeitgeist Research)")
    print(f"{'='*50}")
    post_outlooks = {}
    for camper in staff:
        print(f"\n[{camper.name}] Updated perspective on '{topic}':")
        try:
            outlook = await camper.get_personal_outlook(topic)
            post_outlooks[camper.name] = outlook
            # Extract the main response from the outlook
            response = outlook.get('outlook_response', {})
            if isinstance(response, dict):
                # Check if this is a generated outlook response with 'claim'
                if 'claim' in response:
                    claim = response['claim']
                    print(f"  {claim}")
                else:
                    # Fallback to user prompt for non-outlook responses
                    user_prompt = response.get('user', '')
                    print(f"  Prompt: {user_prompt[:200]}..." if len(user_prompt) > 200 else f"  Prompt: {user_prompt}")
            else:
                print(f"  Response: {response}")
        except Exception as e:
            print(f"  Error getting updated outlook: {e}")
            post_outlooks[camper.name] = {'error': str(e)}
    
    # Generate outlook comparison report
    print(f"\n📊 OUTLOOK EVOLUTION ANALYSIS")
    print(f"{'='*50}")
    await generate_outlook_comparison_report(staff, pre_outlooks, post_outlooks, topic)
    
    print(f"\n{'='*60}")
    print("MEETING SUMMARY")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Participants: {', '.join([c.name for c in staff])}")
    print("\nKey Perspectives:")
    for camper in staff:
        print(f"- {camper.name}: Focused on {', '.join(camper.concerns)}")
    
    # Generate HTML report
    print(f"\n📄 GENERATING HTML REPORT")
    print(f"{'='*50}")
    
    # Create discussion summary
    discussion_summary = f"""
    This meeting brought together {len(staff)} hospital staff members to discuss '{topic}'. 
    Each participant provided their initial perspective, conducted zeitgeist research, and shared updated insights.
    The discussion revealed diverse viewpoints from different hospital roles, leading to a comprehensive 
    understanding of the topic from multiple professional perspectives.
    
    Key participants: {', '.join([f"{c.name} ({c.role})" for c in staff])}
    """
    
    # Run action planning stage
    print(f"\n🚀 PROCEEDING TO ACTION PLANNING")
    print(f"{'='*50}")
    action_proposals, action_items = await run_action_planning_stage(
        topic=topic,
        staff=staff,
        discussion_summary=discussion_summary.strip(),
        research_insights=research_insights
    )
    
    # Generate HTML content with action items
    html_content = generate_html_report(
        topic=topic,
        staff=staff,
        pre_outlooks=pre_outlooks,
        post_outlooks=post_outlooks,
        research_insights=research_insights,
        discussion_summary=discussion_summary.strip(),
        action_proposals=action_proposals,
        action_items=action_items
    )
    
    # Create safe filename and save report
    safe_filename = sanitize_filename(topic)
    report_filename = f"hospital_meeting_{safe_filename}.html"
    report_path = report_filename  # Save in current directory
    
    # Write HTML report
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML report saved: {report_path}")
        print(f"   You can open this file in your browser to view the complete meeting summary with action items.")
    except Exception as e:
        print(f"❌ Error saving HTML report: {e}")
    
    return report_path


async def run_scenario_night_shift_chaos():
    """Scenario: Sarah vents about night-shift chaos, sparking solutions"""
    
    print(f"\n{'='*60}")
    print("SCENARIO: Night Shift Chaos")
    print(f"{'='*60}")
    
    sarah = SarahHeadNurse(party_box=None, config={})
    liam = LiamITSpecialist(party_box=None, config={})
    tom = TomAdminCoordinator(party_box=None, config={})
    priya = PriyaPatientAdvocate(party_box=None, config={})
    elena = DrElenaWardManager(party_box=None, config={})
    
    print("\n💬 Sarah starts venting about last night's chaos...")
    print("[Sarah]: *exhausted* Last night was a disaster. We had three nurses call in sick, ")
    print("         two emergency admits, and the paper logs were a mess. I spent 2 hours ")
    print("         just trying to figure out who gave what medication when!")
    
    print("\n🔍 This sparks Liam to research digital logging solutions...")
    
    # Liam researches digital logging
    research = await liam.research_topic("digital nursing logs hospital medication tracking")
    insight = await liam.share_insight("digital nursing logs", research)
    print(f"\n{insight}")
    
    print("\n💡 Liam proposes a solution...")
    print("[Liam]: What if we implemented a digital log system? I could build something ")
    print("        with QR codes for quick patient ID, voice-to-text for fast entries, ")
    print("        and real-time sync across all devices. No more paper chaos!")
    
    # Others research and respond
    topics_and_researchers = [
        ("digital nursing workflow efficiency", tom),
        ("patient safety medication errors digital systems", priya),
        ("hospital digital transformation budget ROI", elena)
    ]
    
    for topic, researcher in topics_and_researchers:
        print(f"\n🔍 {researcher.name} researches '{topic}'...")
        research = await researcher.research_topic(topic)
        insight = await researcher.share_insight(topic, research)
        print(f"\n{insight}")


async def main():
    """Main demo function"""
    
    print("HOSPITAL ADMINISTRATION ZEITGEIST DEMO")
    print("=" * 50)
    print("Five hospital staff members use Zeitgeist to research and discuss")
    print("patient administration topics from their unique perspectives.")
    
    # Run different scenarios
    scenarios = [
        "patient wait time reduction strategies",
        "hospital staff scheduling optimization",
        "digital patient intake systems",
    ]
    
    generated_reports = []
    for topic in scenarios:
        report_path = await run_hospital_discussion(topic)
        if report_path:
            generated_reports.append(report_path)
        await asyncio.sleep(1)  # Brief pause between discussions
    
    # Run the special night shift chaos scenario
    await run_scenario_night_shift_chaos()
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print("This demo showed how different hospital roles use Zeitgeist to:")
    print("- Research topics from their unique professional perspectives")
    print("- Generate role-specific insights and concerns")
    print("- Collaborate on solutions while maintaining their distinct viewpoints")
    print("\nEach camper's personality shaped their research focus and responses,")
    print("creating a realistic multi-perspective discussion.")
    
    # Display generated HTML reports
    if generated_reports:
        print(f"\n📄 HTML REPORTS GENERATED:")
        print(f"{'='*40}")
        for report_path in generated_reports:
            print(f"📋 {report_path}")
        print(f"\nOpen these files in your browser to view detailed meeting summaries!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())