export type Category = 'All' | 'Frontend' | 'Full-Stack' | 'AI / ML' | 'UI/UX';
export type ProjectType = 'Personal' | 'Class/Team';

export interface Project {
  id: string;
  title: string;
  description: string;
  category: Category;
  projectType: ProjectType;
  image: string;
  tags: string[];
  link?: string;
  github?: string;
}

export interface Experience {
  id: string;
  role: string;
  company: string;
  date: string;
  description: string;
}

export const categories: Category[] = ['All', 'Frontend', 'Full-Stack', 'AI / ML', 'UI/UX'];

export const experience: Experience[] = [
  {
    id: 'edu-1',
    role: 'B.S. Computer Science',
    company: 'University of California, Santa Barbara',
    date: 'Expected Jun 2026',
    description: 'Coursework: Data Structures & Algorithms, Artificial Intelligence, Machine Learning.'
  },
  {
    id: 'exp-1',
    role: 'Frontend Developer Intern',
    company: 'California Seismic — Berkeley',
    date: 'Jun 2025 – Aug 2025',
    description: 'Built and maintained production React components for a public-facing civil engineering exam-prep platform. Implemented dynamic SVG-based visualizations supporting real-time user input. Integrated Plotly.js to generate interactive charts for seismic force calculations.'
  },
  {
    id: 'exp-2',
    role: 'Software Engineering Research Intern',
    company: 'NetFlex / UCSB SNL Lab — Goleta, CA',
    date: 'Sept 2024 – Jun 2025',
    description: 'Designed Python-based data pipelines analyzing latency, throughput, and packet loss. Built LLM-powered RAG workflows translating low-level metrics into user-facing explanations. Co-authored a peer-reviewed paper accepted at the IMC 2025 PRIME Workshop.'
  },
  {
    id: 'exp-3',
    role: 'CSSI Intern',
    company: 'Google — Remote',
    date: 'Jul 2022 – Aug 2022',
    description: 'Completed an intensive SWE program focused on JavaScript, HTML, and CSS. Built interactive web applications and presented a final project.'
  }
];

export const projects: Project[] = [
  {
    id: 'proj-1',
    title: 'NetFlex',
    description: 'Collaborated on the design and development of a Python + React application that evaluates home network performance for non-technical users. Built a RAG-based pipeline that ingests user network measurements (e.g., Ookla, M-Lab) and generates clear, actionable explanations and improvement recommendations.',
    category: 'AI / ML',
    projectType: 'Class/Team',
    image: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?q=80&w=2000&auto=format&fit=crop',
    tags: ['Python', 'React', 'LLMs', 'RAG'],
    github: '#',
  },
  {
    id: 'proj-2',
    title: 'KIT (Kitchen Inventory Tracker)',
    description: 'Designed and built, as a team, a full-stack mobile application with React frontend and FastAPI + Supabase backend. Designed RESTful APIs to manage households, inventory state, recipe calls, and weekly usage summaries.',
    category: 'Full-Stack',
    projectType: 'Class/Team',
    image: 'https://images.unsplash.com/photo-1556910103-1c02745aae4d?q=80&w=2000&auto=format&fit=crop',
    tags: ['React', 'FastAPI', 'Supabase', 'REST API'],
    github: 'https://github.com/ucsb-cs184-w26/team12-KIT',
  }
];

export const skills = [
  { category: 'Languages', items: ['Python', 'C++', 'JavaScript', 'HTML', 'CSS'] },
  { category: 'Frameworks & Libraries', items: ['React', 'Tailwind CSS', 'Plotly.js'] },
  { category: 'Systems & Tools', items: ['Git', 'Linux', 'OpenGL'] },
  { category: 'Data & ML', items: ['LLMs', 'RAG', 'Network Measurement Analysis'] },
  { category: 'Design', items: ['Figma'] },
];

export const hobbies = [
  "🍵 Matcha Enthusiast",
  "🏐 Volleyball",
  "📺 K-Dramas & C-Dramas",
  "🎁 Blind Boxes",
  "💻 Frontend Dev",
  "🎨 UI/UX Design",
];
