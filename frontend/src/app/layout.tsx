import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import "./globals.css";

const poppins = Poppins({ 
  subsets: ["latin"],
  weight: ["400", "700"] 
});

export const metadata: Metadata = {
  title: "似ている有名人診断",
  description: "あなたの顔がどの有名人に似ているか診断します。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body className={poppins.className}>{children}</body>
    </html>
  );
}
