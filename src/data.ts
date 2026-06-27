export type Category = 'All' | 'Frontend' | 'Full-Stack' | 'AI / ML' | 'Other';
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

export const categories: Category[] = ['All', 'Frontend', 'Full-Stack', 'AI / ML', 'Other'];

export const experience: Experience[] = [
  {
    id: 'exp-0',
    role: 'Research Intern',
    company: 'Lawrence Berkeley National Lab (LBNL) — Berkeley',
    date: 'June 2026 – Present',
    description: 'Designing an Agentic Visual RAG for automating the research process in the materials science domain.'
  },
  {
    id: 'edu-1',
    role: 'B.S. Computer Science',
    company: 'University of California, Santa Barbara',
    date: 'Aug 2022 - Jun 2026',
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
    role: 'Network Research Intern',
    company: 'NetFlex / UCSB SNL Lab — Goleta, CA',
    date: 'Sept 2024 – Jun 2025',
    description: 'Designed Python-based data pipelines analyzing latency, throughput, and packet loss. Built LLM-powered RAG workflows translating low-level metrics into user-facing explanations. Co-authored a peer-reviewed paper accepted at the IMC 2025 PRIME Workshop.'
  },
  {
    id: 'exp-3',
    role: 'CSSI Intern',
    company: 'Google, Remote',
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
  },
  {
    id: 'proj-2',
    title: 'KIT (Kitchen Inventory Tracker)',
    description: 'Designed and built, as a team, a full-stack mobile application with React frontend and FastAPI + Supabase backend. Designed RESTful APIs to manage households, inventory state, recipe calls, and weekly usage summaries.',
    category: 'Full-Stack',
    projectType: 'Class/Team',
    image: 'https://images.unsplash.com/photo-1556910103-1c02745aae4d?q=80&w=2000&auto=format&fit=crop',
    tags: ['React', 'FastAPI', 'Supabase', 'Mobile'],
  },
  {
    id: 'proj-3',
    title: 'Momentum',
    description: 'Designed interactive user interfaces in Figma for a fitness tracking application, producing wireframes, prototypes, and design systems for developer handoff.',
    category: 'Other',
    projectType: 'Class/Team',
    image: 'https://americanfitnesscenter.net/assets/hardcore_gym_weights_section-BLWOZhkI.png?q=80&w=2000&auto=format&fit=crop',
    tags: ['Figma', 'UI/UX', 'Miro', 'High-Fidelity'],
    github: 'https://github.com/tizerk/momentum',
    link: 'https://tizerk.github.io/momentum/',
  },
  {
    id: 'proj-4',
    title: 'California Seismic',
    description: 'Contributed to a seismic engineering web application using React, TypeScript, and Git, implementing front-end features, fixing bugs, and participating in code reviews.',
    category: 'Frontend',
    projectType: 'Class/Team',
    image: 'https://earthquake.usgs.gov/monitoring/nsmp/buildings/img/schematic-sdva-2x.png?q=80&w=2000&auto=format&fit=crop',
    tags: ['React', 'Typescript', 'Plotly', 'AWS'],
    github: 'https://github.com/SeijDeLeon/California-Seismic',
    link: 'https://californiaseismic.com/',
  },
  {
    id: 'proj-5',
    title: 'On the Clock',
    description: 'Designed and developed a Unity-based delivery service simulation game featuring package pickup, route planning, and time-based gameplay.',
    category: 'Other',
    projectType: 'Class/Team',
    image: 'https://cdn.phototourl.com/free/2026-06-27-3672c8e4-f967-4c3b-adcb-a793d59a6752.png?q=80&w=2000&auto=format&fit=crop',
    tags: ['Unity', 'Game Development'],
  }
];

export const skills = [
  { category: 'Languages', items: ['Python', 'C++', 'JavaScript', 'HTML', 'CSS'] },
  { category: 'Frameworks & Libraries', items: ['React', 'Tailwind CSS', 'Plotly.js'] },
  { category: 'Systems & Tools', items: ['Git', 'OpenGL'] },
  { category: 'Data & ML', items: ['LLMs', 'RAG', 'Network Measurement Analysis'] },
  { category: 'Design', items: ['Figma'] },
];

export const hobbies = [
  "🍵 Matcha Enthusiast",
  "☕ Cafe Hopping",
  "📺 K-Dramas & C-Dramas",
  "🏐 Volleyball",
  "🎁 Blind Boxes",
  "💻 Frontend Dev",
  "🎨 UI/UX Design",
];
