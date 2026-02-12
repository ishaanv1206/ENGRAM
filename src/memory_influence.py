"""
Memory Influence Layer for the Cognitive Memory Controller.

This module implements the Memory Influence Layer, which converts retrieved
memories into structured conditioning format for LLM context injection.
It formats memories by category, ensures no duplicate information, and
enforces a 500 token budget limit.
"""

from typing import Dict, List
from src.models import MemoryNode, MemoryCategory, RetrievalResult


class MemoryInfluenceLayer:
    """
    Converts retrieved memories into structured LLM context injection.
    
    The layer formats memories by category (pinned, critical, episodic, relational),
    deduplicates information, and enforces a 500 token budget limit.
    """
    
    def __init__(self, max_tokens: int = 500):
        """
        Initialize the Memory Influence Layer.
        
        Args:
            max_tokens: Maximum token overhead for memory injection (default: 500)
        """
        self.max_tokens = max_tokens
        # Rough approximation: 1 token ≈ 4 characters
        self.chars_per_token = 4
        self.max_chars = max_tokens * self.chars_per_token
    
    def inject(self, retrieval: RetrievalResult, query: str) -> str:
        """
        Convert memories to LLM context injection.
        
        Args:
            retrieval: RetrievalResult containing pinned memories and retrieved nodes
            query: The user query (for context, not currently used)
        
        Returns:
            Formatted memory context string ready for LLM injection
        """
        sections = []
        
        # 1. Pinned memories (always first, highest priority)
        if retrieval.pinned:
            sections.append(self._format_pinned(retrieval.pinned))
        
        # 2. Separate memories by category
        critical = [m for m in retrieval.memories if m.category == MemoryCategory.CRITICAL]
        episodic = [m for m in retrieval.memories if m.category == MemoryCategory.EPISODIC]
        relational = [m for m in retrieval.memories if m.category == MemoryCategory.RELATIONAL]
        
        # 3. Format each category
        if critical:
            sections.append(self._format_critical(critical))
        if episodic:
            sections.append(self._format_episodic(episodic))
        if relational:
            sections.append(self._format_relational(relational))
        
        # 4. Combine and truncate to budget
        combined = "\n\n".join(sections)
        return self._truncate_to_budget(combined)
    
    def _format_pinned(self, pinned: Dict[str, str]) -> str:
        """
        Format pinned memories as system constraints.
        
        Args:
            pinned: Dictionary mapping category to content
        
        Returns:
            Formatted pinned memory section
        """
        return f"""<system_constraints>
Language: {pinned.get('language', 'English')}
Style: {pinned.get('style', 'Professional')}
Safety: {pinned.get('safety', 'Standard safety guidelines')}
Timezone: {pinned.get('timezone', 'UTC')}
Persona: {pinned.get('persona', 'AI Assistant')}
</system_constraints>"""
    
    def _format_critical(self, memories: List[MemoryNode]) -> str:
        """
        Format critical memories as facts and constraints.
        
        Args:
            memories: List of critical memory nodes
        
        Returns:
            Formatted critical memory section
        """
        items = []
        seen_content = set()  # Track content to avoid duplicates
        
        for m in memories:
            data = m.structured_data
            
            # Extract preferences
            if 'preferences' in data:
                for pref in data['preferences']:
                    if pref not in seen_content:
                        items.append(f"- User prefers: {pref}")
                        seen_content.add(pref)
            
            # Extract constraints
            if 'constraints' in data:
                for constraint in data['constraints']:
                    if constraint not in seen_content:
                        items.append(f"- Constraint: {constraint}")
                        seen_content.add(constraint)
            
            # Extract commitments
            if 'commitments' in data:
                for commitment in data['commitments']:
                    if commitment not in seen_content:
                        items.append(f"- Commitment: {commitment}")
                        seen_content.add(commitment)
            
            # Extract facts
            if 'facts' in data:
                for fact in data['facts']:
                    if fact not in seen_content:
                        items.append(f"- Fact: {fact}")
                        seen_content.add(fact)
        
        if not items:
            return ""
        
        return f"""<critical_memory>
{chr(10).join(items)}
</critical_memory>"""
    
    def _format_episodic(self, memories: List[MemoryNode]) -> str:
        """
        Format episodic memories as context.
        
        Args:
            memories: List of episodic memory nodes
        
        Returns:
            Formatted episodic memory section
        """
        items = []
        seen_content = set()  # Track content to avoid duplicates
        
        for m in memories:
            # Use the raw content with timestamp
            timestamp = m.created_at.strftime("%Y-%m-%d")
            entry = f"[{timestamp}] {m.content}"
            
            if m.content not in seen_content:
                items.append(f"- {entry}")
                seen_content.add(m.content)
        
        if not items:
            return ""
        
        return f"""<relevant_context>
{chr(10).join(items)}
</relevant_context>"""
    
    def _format_relational(self, memories: List[MemoryNode]) -> str:
        """
        Format relationships as entity graph.
        
        Args:
            memories: List of relational memory nodes
        
        Returns:
            Formatted relational memory section
        """
        items = []
        seen_content = set()  # Track content to avoid duplicates
        
        for m in memories:
            data = m.structured_data
            
            # Extract relationships
            if 'relationships' in data:
                for rel in data['relationships']:
                    if rel not in seen_content:
                        items.append(f"- {rel}")
                        seen_content.add(rel)
            
            # Also include the raw content if it's not already covered
            if m.content not in seen_content and 'relationships' not in data:
                items.append(f"- {m.content}")
                seen_content.add(m.content)
        
        if not items:
            return ""
        
        return f"""<entity_relationships>
{chr(10).join(items)}
</entity_relationships>"""
    
    def _truncate_to_budget(self, text: str) -> str:
        """
        Truncate text to enforce 500 token limit.
        
        Uses character-based approximation (1 token ≈ 4 characters).
        Truncates at section boundaries when possible to maintain structure.
        
        Args:
            text: The formatted memory context
        
        Returns:
            Truncated text within token budget
        """
        if len(text) <= self.max_chars:
            return text
        
        # Try to truncate at section boundaries
        sections = text.split('\n\n')
        result = []
        current_length = 0
        
        for section in sections:
            section_length = len(section) + 2  # +2 for \n\n separator
            
            if current_length + section_length <= self.max_chars:
                result.append(section)
                current_length += section_length
            else:
                # If we can't fit the whole section, try to fit part of it
                remaining = self.max_chars - current_length
                if remaining > 100:  # Only add partial section if we have reasonable space
                    # Truncate at line boundary within the section
                    lines = section.split('\n')
                    partial_section = []
                    partial_length = 0
                    
                    # Check if this section has opening/closing tags
                    has_opening_tag = lines and lines[0].startswith('<')
                    closing_tag = None
                    if has_opening_tag and lines[0].startswith('<') and '>' in lines[0]:
                        tag_name = lines[0][1:lines[0].index('>')]
                        closing_tag = f"</{tag_name}>"
                    
                    for line in lines:
                        line_length = len(line) + 1  # +1 for \n
                        # Reserve space for closing tag if needed
                        space_needed = line_length
                        if closing_tag and line != closing_tag:
                            space_needed += len(closing_tag) + 1
                        
                        if partial_length + space_needed <= remaining:
                            partial_section.append(line)
                            partial_length += line_length
                        else:
                            break
                    
                    # Add closing tag if we have an opening tag and didn't include the closing tag
                    if closing_tag and partial_section and closing_tag not in partial_section:
                        if partial_length + len(closing_tag) + 1 <= remaining:
                            partial_section.append(closing_tag)
                    
                    if partial_section:
                        result.append('\n'.join(partial_section))
                
                break  # Stop adding more sections
        
        return '\n\n'.join(result)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a given text.
        
        Uses rough approximation: 1 token ≈ 4 characters.
        
        Args:
            text: The text to estimate
        
        Returns:
            Estimated token count
        """
        return len(text) // self.chars_per_token
